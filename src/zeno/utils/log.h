#pragma once

#include <zeno/utils/api.h>
#include <zeno/utils/source_location.h>
#include <zeno/utils/cformat.h>
#ifdef ZENO_ENABLE_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/core.h>
#else
#include <zeno/utils/format.h>
#endif
#include <string_view>

namespace zeno {

template <class T>
class __with_source_location {
    T m_value;
    source_location m_loc;

public:
    template <class Arg>
    __with_source_location(Arg &&arg, source_location loc = source_location::current())
        : m_value(std::forward<Arg>(arg)), m_loc(loc) {}

    operator auto const &() const { return m_value; }
    operator auto &() { return m_value; }
    auto const &value() const { return m_value; }
    auto &value() { return m_value; }
    auto const &location() const { return m_loc; }
    auto &location() { return m_loc; }
};

#ifdef ZENO_ENABLE_SPDLOG
ZENO_API spdlog::logger *__get_spdlog_logger();

namespace log_level = spdlog::level;

static void set_log_level(log_level::level_enum level) {
    __get_spdlog_logger()->set_level(level);
}

template <class ...Args>
void log_print(log_level::level_enum level, __with_source_location<std::string_view> const &msg, Args &&...args) {
    spdlog::source_loc loc(msg.location().file_name(), msg.location().line(), msg.location().function_name());
    __get_spdlog_logger()->log(loc, level, fmt::format(msg.value(), std::forward<Args>(args)...));
}

#else
namespace log_level {
enum level_enum { trace, debug, info, critical, warn, err };
};

ZENO_API void set_log_level(log_level::level_enum level);
ZENO_API void __impl_log_print(log_level::level_enum level, source_location const &loc, std::string_view msg);

template <class ...Args>
void log_print(log_level::level_enum level, __with_source_location<std::string_view> const &msg, Args &&...args) {
    __impl_log_print(level, msg.location(), format(msg.value(), std::forward<Args>(args)...));
}
#endif

#define _PER_LOG_LEVEL(x, y) \
template <class ...Args> \
void log_##x(__with_source_location<std::string_view> const &msg, Args &&...args) { \
    log_print(log_level::y, msg, std::forward<Args>(args)...); \
} \
template <class ...Args> \
void log_##x##f(__with_source_location<const char *> const &msg, Args &&...args) { \
    log_print(log_level::y, "{}", cformat(msg, std::forward<Args>(args)...)); \
}
_PER_LOG_LEVEL(trace, trace)
_PER_LOG_LEVEL(debug, debug)
_PER_LOG_LEVEL(info, info)
_PER_LOG_LEVEL(critical, critical)
_PER_LOG_LEVEL(warn, warn)
_PER_LOG_LEVEL(error, err)
#undef _PER_LOG_LEVEL

}
