#pragma once

#include <zeno/utils/api.h>
#include <zeno/utils/source_location.h>
#include <zeno/utils/cformat.h>
#include <zeno/utils/format.h>
#include <string_view>

namespace zeno {

template <class T>
class __with_source_location {
    T m_value;
    source_location m_loc;

public:
    template <class U>
    __with_source_location(U &&u, source_location loc = source_location::current())
        : m_value(std::forward<U>(u)), m_loc(loc) {}

    auto const &value() const { return m_value; }
    auto &value() { return m_value; }
    auto const &location() const { return m_loc; }
    auto &location() { return m_loc; }
};

enum class log_level_t {
    trace, debug, info, critical, warn, error,
};

ZENO_API void set_log_level(log_level_t level);
ZENO_API void set_log_stream(std::ostream &osin);
ZENO_API bool __check_log_level(log_level_t level);
ZENO_API void __impl_log_print(log_level_t level, source_location const &loc, std::string_view msg);

template <class ...Args>
void log_print(log_level_t level, __with_source_location<std::string_view> const &msg, Args &&...args) {
    if (__check_log_level(level))
        __impl_log_print(level, msg.location(), format(msg.value(), std::forward<Args>(args)...));
}

#define _ZENO_PER_LOG_LEVEL(x) \
template <class ...Args> \
void log_##x(__with_source_location<std::string_view> const &msg, Args &&...args) { \
    log_print(log_level_t::x, msg, std::forward<Args>(args)...); \
} \
template <class ...Args> \
void log_##x##f(__with_source_location<const char *> const &msg, Args &&...args) { \
    log_print(log_level_t::x, {"{}", msg.location()}, cformat(msg.value(), std::forward<Args>(args)...)); \
}
_ZENO_PER_LOG_LEVEL(trace)
_ZENO_PER_LOG_LEVEL(debug)
_ZENO_PER_LOG_LEVEL(info)
_ZENO_PER_LOG_LEVEL(critical)
_ZENO_PER_LOG_LEVEL(warn)
_ZENO_PER_LOG_LEVEL(error)
#undef _ZENO_PER_LOG_LEVEL

}
