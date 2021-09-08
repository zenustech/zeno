#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <zeno/utils/initializer_list.h>
#include <zeno/utils/logger.h>
#include <cassert>
#include <cstring>
#include <cstdio>

namespace zeno {

std::shared_ptr<spdlog::logger> g_logger;

ZENO_API spdlog::logger &logger() {
    return *g_logger;
}

auto level_from_str(const char *str, const char *defl) {
    if (!str) str = defl;
    if (0) {
#define _PER_LEVEL(x, y) } else if (!strcmp(str, #x)) { return spdlog::level::y;
    _PER_LEVEL(t, trace)
    _PER_LEVEL(d, debug)
    _PER_LEVEL(i, info)
    _PER_LEVEL(c, critical)
    _PER_LEVEL(w, warn)
    _PER_LEVEL(e, err)
#undef _PER_LEVEL
    } else {
        return spdlog::level::info;
    }
}

static void initialize_spdlog() {
    std::vector<spdlog::sink_ptr> sinks;
    if (!getenv("ZEN_NO_LOG_CONSOLE")) {
        auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink->set_level(level_from_str(getenv("ZEN_LOG_LEVEL"), "info"));
        sinks.push_back(std::move(sink));
    }
    if (auto logfile = getenv("ZEN_LOG_FILE"); logfile) {
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logfile, true);
        sink->set_level(level_from_str(getenv("ZEN_FILE_LOG_LEVEL"), "trace"));
        sinks.push_back(std::move(sink));
    }
    g_logger = std::make_shared<spdlog::logger>("zeno", to_initializer_list(sinks));
}

static int initialize_spdlog_helper = (initialize_spdlog(), 0);

}
