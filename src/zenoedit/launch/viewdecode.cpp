#ifdef ZENO_MULTIPROCESS
#include "viewdecode.h"
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
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
        zeno::getSession().globalState->addViewObject(object);

    } else {
        zeno::log_warn("unknown packet action type {}", action);
        return false;
    }
    return true;
}

struct Header {
    size_t total_size;
    size_t info_size;
};

bool parsePacket(const char *buf, Header const &header) {
    zeno::log_info("viewDecodePacket: {}", std::string(buf, header.total_size));
    if (header.total_size < header.info_size) {
        zeno::log_warn("total_size < info_size");
        return false;
    }

    rapidjson::Document doc;
    doc.Parse(buf, header.info_size);

    if (!doc.IsObject()) {
        zeno::log_warn("document root not object");
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

    // encode rule: \a, \b, then 8-byte of SIZE, then the SIZE-byte of DATA
    void append(const char *buf, size_t n)
    {
        for (auto p = buf; p < buf + n; p++) {
            if (phase == 3) {
                buffer[buffercurr++] = *p;
                if (buffercurr >= header().total_size) {
                    parsePacket(buffer.data(), header());
                    phase = 0;
                }
            } else if (phase == 0) {
                if (*p == '\a') {
                    phase = 1;
                }
            } else if (phase == 1) {
                if (*p == '\b') {
                    phase = 2;
                }
            } else if (phase == 2) {
                headerbuf[headercurr++] = *p;
                if (headercurr >= sizeof(headerbuf)) {
                    headercurr = 0;
                    phase = 3;
                    buffer.resize(header().total_size);
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
}

void viewDecodeAppend(const char *buf, size_t n)
{
    zeno::log_debug("viewDecodeAppend n={}", n);
    viewDecodeData.append(buf, n);
}
#endif
