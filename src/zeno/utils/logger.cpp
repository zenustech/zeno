#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <zeno/utils/logger.h>
#include <cassert>
#include <cstring>
#include <cstdio>

namespace zeno {

std::shared_ptr<spdlog::logger> g_logger;

ZENO_API spdlog::logger *get_spdlog_logger() {
    assert(g_logger);
    return g_logger.get();
}

static void initialize_spdlog() {
    spdlog::set_pattern("%^[%L %X.%e] (%s:%#) %v%$");
    if (auto env = getenv("ZENO_LOGLEVEL"); env) {
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
    if (auto env = getenv("ZENO_LOGFILE"); env) {
        g_logger = spdlog::basic_logger_mt("zeno", env);
    } else {
        g_logger = spdlog::stderr_color_mt("zeno");
    }
}

static int initialize_spdlog_helper = (initialize_spdlog(), 0);

}
