#include <zeno/utils/log.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#ifdef ZENO_ENABLE_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#else
#include <zeno/utils/format.h>
#include <zeno/utils/ansiclr.h>
#include <zeno/utils/arrayindex.h>
#include <iostream>
#include <chrono>
#endif

namespace zeno {

#ifndef ZENO_ENABLE_SPDLOG
static log_level::level_enum curr_level = log_level::info;
static std::ostream *os = &std::cerr;

ZENO_API void set_log_level(log_level::level_enum level) {
    curr_level = level;
}

ZENO_API bool __check_log_level(log_level::level_enum level) {
    return level >= curr_level;
}

ZENO_API void set_log_stream(std::ostream &osin) {
    os = &osin;
}

ZENO_API void __impl_log_print(log_level::level_enum level, source_location const &loc, std::string_view msg) {
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
}
#endif

namespace { struct LogInitializer {
#ifdef ZENO_ENABLE_SPDLOG
    std::shared_ptr<spdlog::logger> g_logger;
#endif

    LogInitializer() {
#ifdef ZENO_ENABLE_SPDLOG
        if (auto env = std::getenv("ZENO_LOGFILE"); env) {
            g_logger = spdlog::basic_logger_mt("zeno", env);
        } else {
            g_logger = spdlog::stderr_color_mt("zeno");
        }
        g_logger->set_pattern("%^[%L %X.%e] (%g:%#) %v%$");
#endif

        if (auto env = std::getenv("ZENO_LOGLEVEL"); env) {
            if (0) {
#ifdef ZENO_ENABLE_SPDLOG
#define _PER_LEVEL(x, y) } else if (!std::strcmp(env, #x)) { g_logger->set_level(log_level::y);
#else
#define _PER_LEVEL(x, y) } else if (!std::strcmp(env, #x)) { set_log_level(log_level::y);
#endif
            _PER_LEVEL(trace, trace)
            _PER_LEVEL(debug, debug)
            _PER_LEVEL(info, info)
            _PER_LEVEL(critical, critical)
            _PER_LEVEL(warn, warn)
            _PER_LEVEL(error, err)
#undef _PER_LEVEL
            }
        }


#if defined(__DATE__) && defined(__TIME__)
#ifdef ZENO_ENABLE_SPDLOG
        auto loc = source_location::current();
        g_logger->log(spdlog::source_loc(loc.file_name(), loc.line(), loc.function_name()),
                      spdlog::level::info, "build date: {} {}", __DATE__, __TIME__);
#else
        log_info("build date: {} {}", __DATE__, __TIME__);
#endif
#endif
    }
}; }

#ifdef ZENO_ENABLE_SPDLOG
ZENO_API spdlog::logger *__get_spdlog_logger() {
    static LogInitializer init;
    return init.g_logger.get();
}
#else
static LogInitializer g_log_init;
#endif

}
