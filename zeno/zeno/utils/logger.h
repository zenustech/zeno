#pragma once

#include <zeno/utils/api.h>
#include <zeno/utils/source_location.h>
#include <string_view>
#include <spdlog/spdlog.h>
#include <string_view>
#include <memory>

namespace zeno {

ZENO_API spdlog::logger *get_spdlog_logger();

namespace __log_print {
    struct format_string {
        std::string_view str;
        source_location loc;

        format_string(const char *str, source_location loc = source_location::current())
            : str(str), loc(loc) {}
        format_string(std::string_view str, source_location loc = source_location::current())
            : str(str), loc(loc) {}
    };
};

template <class ...Args>
void log_print(spdlog::level_t log_level,
        __log_print::format_string const &fmt, Args &&...args) {
    spdlog::source_loc loc(fmt.loc.file_name(), fmt.loc.line(), fmt.loc.function_name());
    get_spdlog_logger()->log(loc, log_level, fmt.str, std::forward<Args>(args)...);
}

#define _PER_LOG_LEVEL(x, y) \
template <class ...Args> \
void log_##x(__log_print::format_string const &fmt, Args &&...args) { \
    log_print(spdlog::level::y, fmt, std::forward<Args>(args)...); \
}
_PER_LOG_LEVEL(trace, trace)
_PER_LOG_LEVEL(debug, debug)
_PER_LOG_LEVEL(info, info)
_PER_LOG_LEVEL(critical, critical)
_PER_LOG_LEVEL(warn, warn)
_PER_LOG_LEVEL(error, err)
#undef _PER_LOG_LEVEL

}
