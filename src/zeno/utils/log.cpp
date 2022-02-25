#ifdef ZENO_ENABLE_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <zeno/utils/logger.h>
#include <cassert>
#include <cstring>
#include <cstdio>

namespace zeno {

std::shared_ptr<spdlog::logger> g_logger;

ZENO_API spdlog::logger *__get_spdlog_logger() {
    assert(g_logger);
    return g_logger.get();
}

namespace {
struct initialize_spdlog {
    initialize_spdlog() {
        spdlog::set_pattern("%^[%L %X.%e] (%g:%#) %v%$");
        if (auto env = std::getenv("ZENO_LOGLEVEL"); env) {
            if (0) {
#define _PER_LEVEL(x, y) } else if (!strcmp(env, #x)) { spdlog::set_level(spdlog::level::y);
            _PER_LEVEL(trace, trace)
            _PER_LEVEL(debug, debug)
            _PER_LEVEL(info, info)
            _PER_LEVEL(critical, critical)
            _PER_LEVEL(warn, warn)
            _PER_LEVEL(error, err)
#undef _PER_LEVEL
            }
        }
        if (auto env = std::getenv("ZENO_LOGFILE"); env) {
            g_logger = spdlog::basic_logger_mt("zeno", env);
        } else {
            g_logger = spdlog::stderr_color_mt("zeno");
        }
#if defined(__DATE__) && defined(__TIME__)
        zeno::log_info("build date: {} {}", __DATE__, __TIME__);
#endif
#if defined(__GNUC__)
        zeno::log_info("compiler: gcc version {}", __GNUC__);
#elif defined(__clang__)
        zeno::log_info("compiler: clang version {}", __clang__);
#elif defined(_MSC_VER)
        zeno::log_info("compiler: msvc version {}", _MSC_VER);
#endif
    }
} initialize_spdlog_v;
}

}
#endif
