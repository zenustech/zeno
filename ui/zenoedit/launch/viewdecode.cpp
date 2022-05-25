#ifdef ZENO_MULTIPROCESS
#include "viewdecode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zeno/utils/log.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/funcs/ObjectCodec.h>
#include <rapidjson/document.h>
#include <type_traits>
#include <iostream>
#include <cassert>
#include <vector>
#include <string>

namespace {

struct Header {
    size_t total_size;
    size_t info_size;
    size_t magicnum;
    size_t checksum;

    bool isValid() const {
        if (magicnum != 314159265) return false;
        return (total_size ^ info_size ^ magicnum ^ checksum) == 0;
    }
};

struct PacketProc {
    int globalCommNeedClean = 0;
    int globalCommNeedNewFrame = 0;

    void onStart() {
        globalCommNeedClean = 1;
        globalCommNeedNewFrame = 0;
        zeno::getSession().globalState->clearState();
        zeno::getSession().globalStatus->clearState();
        zeno::getSession().globalState->working = true;
    }

    void onFinish() {
        clearGlobalIfNeeded();
        zeno::getSession().globalState->working = false;
    }

    void clearGlobalIfNeeded() {
        if (globalCommNeedClean) {
            zeno::log_debug("PacketProc::clearGlobalStateIfNeeded: globalStateNeedClean");
            zeno::getSession().globalComm->clearState();
            globalCommNeedClean = 0;
        }
        if (globalCommNeedNewFrame) {
            zeno::log_debug("PacketProc::clearGlobalStateIfNeeded: globalCommNeedNewFrame");
            zeno::getSession().globalComm->newFrame();
            globalCommNeedNewFrame = 0;
        }
    }

    bool processPacket(std::string const &action, std::string const &objKey, const char *buf, size_t len) {

        if (action == "viewObject") {
            zeno::log_debug("decoding object");
            auto object = zeno::decodeObject(buf, len);
            //zeno::log_debug("object ident=[{}]", object->userData().get("ident"));
            if (!object) {
                zeno::log_warn("failed to decode view object");
                return false;
            }
            clearGlobalIfNeeded();
            zeno::getSession().globalComm->addViewObject(objKey, object);

            //need to notify the GL to update.
            zenoApp->getMainWindow()->updateViewport();

        } else if (action == "newFrame") {
            globalCommNeedNewFrame = 1; // postpone `zeno::getSession().globalComm->newFrame();`

        } else if (action == "finishFrame") {
            zeno::getSession().globalComm->finishFrame();

        } else if (action == "frameRange") {
            auto pos = objKey.find(':');
            if (pos != std::string::npos) {
                int beg = std::stoi(objKey.substr(0, pos));
                int end = std::stoi(objKey.substr(pos + 1));
                zeno::getSession().globalComm->frameRange(beg, end);
                zeno::getSession().globalState->frameid = beg;
            }

        } else if (action == "reportStatus") {
            std::string statJson{buf, len};
            zeno::getSession().globalStatus->fromJson(statJson);

        } else {
            zeno::log_warn("unknown packet action type {}", action);
            return false;
        }
        return true;
    }

    bool parsePacket(const char *buf, Header const &header) {
        zeno::log_debug("viewDecodePacket: n={}", header.total_size);
        if (header.total_size < header.info_size) {
            zeno::log_warn("total_size < info_size");
            return false;
        }

        rapidjson::Document doc;
        doc.Parse(buf, header.info_size);

        if (!doc.IsObject()) {
            zeno::log_warn("document root not object: {}", std::string(buf, header.info_size));
            return false;
        }
        auto root = doc.GetObject();

        
        std::string action;
        if (auto it = root.FindMember("action"); it != root.MemberEnd() && it->value.IsString()) {
            action.assign(it->value.GetString(), it->value.GetStringLength());
        } else {
            zeno::log_warn("no string entry named 'action'");
            return false;
        }

        std::string objKey;
        if (auto it = root.FindMember("key"); it != root.MemberEnd() && it->value.IsString()) {
            objKey.assign(it->value.GetString(), it->value.GetStringLength());
        }

        const char *data = buf + header.info_size;
        size_t size = header.total_size - header.info_size;

        zeno::log_debug("decoder got action=[{}] key=[{}] size={}", action, objKey, size);

        return processPacket(action, objKey, data, size);
    }

} packetProc;


struct ViewDecodeData {

    std::vector<char> buffer;
    int phase = 0;
    size_t buffercurr = 0;
    size_t headercurr = 0;
    char headerbuf[sizeof(Header)] = {};
    size_t cloglen = 0;
    char clogbuf[4100];

    void clear()
    {
        buffer.clear();
        phase = 0;
        buffercurr = 0;
        headercurr = 0;
        std::memset(headerbuf, 0, sizeof(headerbuf));
    }

    auto const &header() const
    {
        return *(Header const *)headerbuf;
    }

    void finish()
    {
        std::clog.flush();
    }

    // encode rule: \a, \b, \r, \t, then 8-byte of SIZE, then the SIZE-byte of DATA
    void append(const char *buf, size_t n)
    {
        for (auto p = buf; p < buf + n; p++) {
            if (phase == 5) {
#if 1
                size_t rest = std::min(size_t(buf + n - p), header().total_size - buffercurr);
                if (rest) {
                    std::memcpy(buffer.data() + buffercurr, p, rest);
                    p += rest - 1;
                    buffercurr += rest;
                }
#else
                buffer[buffercurr++] = *p;
#endif
                if (buffercurr >= header().total_size) {
                    buffercurr = 0;
                    zeno::log_debug("finish rx, parsing packet of size {}", header().total_size);
                    packetProc.parsePacket(buffer.data(), header());
                    phase = 0;
                }
            } else if (phase == 0) {
                if (*p == '\a') {
                    phase = 1;
                } else {
                    clogbuf[cloglen++] = *p;
                    // clog is captured by luzh log panel
                    if (*p == '\n' || cloglen >= sizeof(clogbuf) - 4) {
                        std::clog << std::string_view(clogbuf, cloglen);
                        //std::ostreambuf_iterator<char> oit(std::clog);
                        //std::copy_n(clogbuf, cloglen, oit);
                        cloglen = 0;
                    }
                }
            } else if (phase == 1) {
                if (*p == '\b') {
                    phase = 2;
                } else {
                    phase = 0;
                }
            } else if (phase == 2) {
                if (*p == '\r') {
                    phase = 3;
                } else {
                    phase = 0;
                }
            } else if (phase == 3) {
                if (*p == '\t') {
                    zeno::log_debug("got abrt sequence, entering phase-4");
                    phase = 4;
                } else {
                    phase = 0;
                }
            } else if (phase == 4) {
                headerbuf[headercurr++] = *p;
                if (headercurr >= sizeof(Header)) {
                    headercurr = 0;
                    phase = 5;
                    zeno::log_debug("got header: total_size={}, info_size={}, magicnum={}", header().total_size, header().info_size, header().magicnum);
                    if (!header().isValid()) {
                        zeno::log_debug("header checksum invalid, giving up");
                        phase = 0;
                    } else {
                        buffer.resize(header().total_size);
                    }
                }
            } else {
                phase = 0;
            }
        }
    }

} viewDecodeData;

}

void viewDecodeFinish()
{
    viewDecodeData.finish();
    packetProc.onFinish();
}

void viewDecodeClear()
{
    zeno::log_debug("viewDecodeClear");
    viewDecodeData.clear();
    packetProc.onStart();
}

void viewDecodeAppend(const char *buf, size_t n)
{
    zeno::log_debug("viewDecodeAppend n={}", n);
    viewDecodeData.append(buf, n);
}
#endif
