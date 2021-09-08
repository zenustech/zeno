#pragma once

#include <zeno/utils/api.h>
#include <zeno/utils/source_location.h>
#include <string_view>
#include <spdlog/spdlog.h>
#include <memory>

namespace zeno {

ZENO_API spdlog::logger *get_spdlog_logger();

template <class ...Args>
void log_print(source_location const &location, spdlog::level_t log_level,
        fmt::format_string<Args...> &&fmt, Args &&...args) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    get_spdlog_logger()->log(loc, log_level,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

template <class ...Args>
void log_trace(fmt::format_string<Args...> &&fmt, Args &&...args,
        source_location const &location = source_location::current()) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    log_print(location, spdlog::level::trace,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

template <class ...Args>
void log_debug(fmt::format_string<Args...> &&fmt, Args &&...args,
        source_location const &location = source_location::current()) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    log_print(location, spdlog::level::debug,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

template <class ...Args>
void log_info(fmt::format_string<Args...> &&fmt, Args &&...args,
        source_location const &location = source_location::current()) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    log_print(location, spdlog::level::info,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

template <class ...Args>
void log_critical(fmt::format_string<Args...> &&fmt, Args &&...args,
        source_location const &location = source_location::current()) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    log_print(location, spdlog::level::critical,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

template <class ...Args>
void log_warn(fmt::format_string<Args...> &&fmt, Args &&...args,
        source_location const &location = source_location::current()) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    log_print(location, spdlog::level::warn,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

template <class ...Args>
void log_error(fmt::format_string<Args...> &&fmt, Args &&...args,
        source_location const &location = source_location::current()) {
    spdlog::source_loc loc(location.file_name(), location.line(), location.function_name());
    log_print(location, spdlog::level::err,
            std::forward<fmt::format_string<Args...>>(fmt),
            std::forward<Args>(args)...);
}

}
