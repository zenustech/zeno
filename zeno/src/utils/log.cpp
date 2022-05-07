#include <zeno/utils/log.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <zeno/utils/format.h>
#include <zeno/utils/ansiclr.h>
#include <zeno/utils/arrayindex.h>
#include <iostream>
#include <chrono>

namespace zeno {

static log_level::level_enum curr_level = log_level::info;
static std::ostream *os = &std::clog;

ZENO_API void set_log_level(log_level::level_enum level) {
    curr_level = level;
}

ZENO_API bool __check_log_level(log_level::level_enum level) {
    return level >= curr_level;
}

ZENO_API void set_log_stream(std::ostream &osin) {
    os = &osin;
}

ZENO_API void __impl_log_print(log_level level, source_location const &loc, std::string_view msg) {
    auto now = std::chrono::steady_clock::now();
    auto sod = std::chrono::floor<std::chrono::duration<int, std::ratio<24 * 60 * 60, 1>>>(now);
    auto mss = std::chrono::floor<std::chrono::milliseconds>(now - sod).count();
    int linlev = (int)level - (int)log_level::trace;

    //auto fgclr = ansiclr::fg[make_array(ansiclr::white, ansiclr::cyan, ansiclr::green,
                                        //ansiclr::cyan | ansiclr::light, ansiclr::yellow | ansiclr::light,
                                        //ansiclr::red | ansiclr::light)[linlev]];
    *os << format("[{} {02d}:{02d}:{02d}.{03d}] ({}:{}) {}\n",
                  "TDICWE"[linlev],
                  mss / 1000 / 60 / 60 % 24, mss / 1000 / 60 % 60, mss / 1000 % 60, mss % 1000,
                  loc.file_name(), loc.line(), msg);
    if (level >= log_level::critical)
        os.flush();
}
#endif

namespace {
struct LogInitializer {
    LogInitializer() {
        if (auto env = std::getenv("ZENO_LOGLEVEL"); env) {
            if (0) {
#define _ZENO_PER_LOG_LEVEL(x, y) } else if (!std::strcmp(env, #x)) { set_log_level(log_level::y);
_PER_LOG_LEVEL(trace)
_PER_LOG_LEVEL(debug)
_PER_LOG_LEVEL(info)
_PER_LOG_LEVEL(critical)
_PER_LOG_LEVEL(warn)
_PER_LOG_LEVEL(error)
#undef _ZENO_PER_LOG_LEVEL
            }
        }
    }
};
static LogInitializer g_log_init;
}

}
