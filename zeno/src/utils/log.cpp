#include <zeno/utils/log.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <zeno/utils/format.h>
//#include <zeno/utils/ansiclr.h>
#include <zeno/utils/arrayindex.h>
#include <iostream>
#include <chrono>

namespace zeno {

static log_level_t curr_level = log_level_t::info;
static std::ostream *os = &std::clog;

ZENO_API void set_log_level(log_level_t level) {
    curr_level = level;
}

ZENO_API bool __check_log_level(log_level_t level) {
    return level >= curr_level;
}

ZENO_API void set_log_stream(std::ostream &osin) {
    os = &osin;
}

ZENO_API void __impl_log_print(log_level_t level, source_location const &loc, std::string_view msg) {
    auto now = std::chrono::steady_clock::now();
    auto sod = std::chrono::floor<std::chrono::duration<int, std::ratio<24 * 60 * 60, 1>>>(now);
    auto mss = std::chrono::floor<std::chrono::milliseconds>(now - sod).count();
    int linlev = (int)level - (int)log_level_t::trace;
    //*os << ansiclr::fg[make_array(ansiclr::white, ansiclr::cyan, ansiclr::green,
                                  //ansiclr::cyan | ansiclr::light, ansiclr::yellow | ansiclr::light,
                                  //ansiclr::red | ansiclr::light)[linlev]];
    *os << format("[{} {02d}:{02d}:{02d}.{03d}] ({}:{}) {}\n",
                  "TDICWE"[linlev],
                  mss / 1000 / 60 / 60 % 24, mss / 1000 / 60 % 60,
                  mss / 1000 % 60, mss % 1000,
                  loc.file_name(), loc.line(),
                  msg);
    //*os << ansiclr::reset;
    if (level >= log_level_t::critical)
        os->flush();
}

namespace {
struct LogInitializer {
    LogInitializer() {
        if (auto env = std::getenv("ZENO_LOGLEVEL"); env) {
            if (0) {
#define _ZENO_PER_LOG_LEVEL(x) } else if (!std::strcmp(env, #x)) { set_log_level(log_level_t::x);
_ZENO_PER_LOG_LEVEL(trace)
_ZENO_PER_LOG_LEVEL(debug)
_ZENO_PER_LOG_LEVEL(info)
_ZENO_PER_LOG_LEVEL(critical)
_ZENO_PER_LOG_LEVEL(warn)
_ZENO_PER_LOG_LEVEL(error)
#undef _ZENO_PER_LOG_LEVEL
            }
        }
    }
};
static LogInitializer g_log_init;
}

}
