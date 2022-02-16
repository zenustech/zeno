#ifdef ZENO_MULTIPROCESS
#include "viewdecode.h"
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/funcs/ObjectCodec.h>
#include <rapidjson/document.h>
#include <type_traits>
#include <cassert>
#include <vector>
#include <string>

namespace {

bool processPacket(std::string const &action, const char *buf, size_t len) {

    if (action == "viewObject") {
        auto object = zeno::decodeObject(buf, len);
        if (!object) {
            zeno::log_warn("failed to decode view object");
            return false;
        }
        zeno::getSession().globalComm->addViewObject(object);

    } else if (action == "newFrame") {
        zeno::getSession().globalComm->newFrame();

    } else {
        zeno::log_warn("unknown packet action type {}", action);
        return false;
    }
    return true;
}

struct Header {
    size_t total_size;
    size_t info_size;
    size_t magicnum;
    size_t checksum;

    bool isValid() const {
        if (magicnum != 314159265) return 0;
        return (total_size ^ info_size ^ magicnum ^ checksum) == 0;
    }
};

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
    auto it = root.FindMember("action");
    if (it == root.MemberEnd() || !it->value.IsString()) {
        zeno::log_warn("no string entry named 'action'");
        return false;
    }
    std::string action{it->value.GetString(), it->value.GetStringLength()};

    const char *data = buf + header.info_size;
    size_t size = header.total_size - header.info_size;

    return processPacket(action, data, size);
}


struct ViewDecodeData {

    std::vector<char> buffer;
    int phase = 0;
    size_t buffercurr = 0;
    size_t headercurr = 0;
    char headerbuf[sizeof(Header)] = {};

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

    // encode rule: \a, \b, \r, \t, then 8-byte of SIZE, then the SIZE-byte of DATA
    void append(const char *buf, size_t n)
    {
        for (auto p = buf; p < buf + n; p++) {
            if (phase == 5) {
                buffer[buffercurr++] = *p;
                if (buffercurr >= header().total_size) {
                    buffercurr = 0;
                    zeno::log_debug("finish rx, parsing packet of size {}", header().total_size);
                    parsePacket(buffer.data(), header());
                    phase = 0;
                }
            } else if (phase == 0) {
                if (*p == '\a') {
                    phase = 1;
                } else {
                    phase = 0;
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

void viewDecodeClear()
{
    zeno::log_debug("viewDecodeClear");
    viewDecodeData.clear();
    zeno::getSession().globalComm->clearState();
}

void viewDecodeAppend(const char *buf, size_t n)
{
    zeno::log_debug("viewDecodeAppend n={}", n);
    viewDecodeData.append(buf, n);
}
#endif
