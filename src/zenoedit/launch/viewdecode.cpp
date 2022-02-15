#ifdef ZENO_MULTIPROCESS
#include "viewdecode.h"
#include <zeno/utils/log.h>
#include <vector>
#include <string>

namespace {

void viewDecodePacket(const char *buf, size_t n) {
}

}

namespace {

std::vector<char> buffer;
int phase = 0;
size_t count = 0;
size_t curr = 0;
size_t countercount = 0;
char counter[sizeof(size_t)] = {};

}

void viewDecodeClear()
{
    phase = 0;
    count = 0;
    curr = 0;
    countercount = 0;
    std::memset(counter, 0, sizeof(counter));
    buffer.clear();
}

// encode rule: \a, then 8-byte of SIZE, then the SIZE-byte of DATA
void viewDecodeAppend(const char *buf, size_t n)
{
    zeno::log_info("viewDecodeAppend: {}", std::string(buf, n));
    for (auto p = buf; p < buf + n; p++) {
        if (phase == 2) {
            buffer[curr++] = *p;
            if (curr >= count) {
                viewDecodePacket(buffer.data(), count);
                phase = 0;
            }
        } else if (phase == 0) {
            if (*p == '\a') {
                phase = 1;
            }
        } else if (phase == 1) {
            counter[countercount - 1] = *p;
            countercount++;
            if (countercount == sizeof(size_t)) {
                countercount = 0;
                phase = 2;
                count = *(size_t *)counter;
                buffer.resize(count);
            }
        } else {
            phase = 0;
        }
    }
}
#endif
