#pragma once

#include <zeno/utils/api.h>
#include <zeno/utils/source_location.h>
#include <zeno/utils/format.h>
#ifdef ZENO_ENABLE_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/core.h>
#endif
#include <string_view>
#include <sstream>
#include <memory>

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

template <class ...Args>
void log_print(log_level::level_enum level, __with_source_location<std::string_view> const &fmt, Args &&...args) {
    spdlog::source_loc loc(fmt.location().file_name(), fmt.location().line(), fmt.location().function_name());
    __get_spdlog_logger()->log(loc, level, fmt::format(fmt.value(), std::forward<Args>(args)...));
}

#else
namespace log_level {
enum level_enum { trace, debug, info, critical, warn, err };
};

template <class ...Args>
void log_print(log_level::level_enum level, __with_source_location<std::string_view> const &fmt, Args &&...args) {
}
#endif

#define _PER_LOG_LEVEL(x, y) \
template <class ...Args> \
void log_##x(__with_source_location<std::string_view> const &fmt, Args &&...args) { \
    log_print(log_level::y, fmt, std::forward<Args>(args)...); \
} \
template <class ...Args> \
void log_##x##f(__with_source_location<const char *> const &fmt, Args &&...args) { \
    log_print(log_level::y, "{}", format(fmt, std::forward<Args>(args)...)); \
}
_PER_LOG_LEVEL(trace, trace)
_PER_LOG_LEVEL(debug, debug)
_PER_LOG_LEVEL(info, info)
_PER_LOG_LEVEL(critical, critical)
_PER_LOG_LEVEL(warn, warn)
_PER_LOG_LEVEL(error, err)
#undef _PER_LOG_LEVEL

inline namespace loggerstd {

static inline constexpr char endl = '\n';

static inline struct __logger_ostream {
    struct __logger_ostream_proxy {
        std::stringstream ss;

        source_location m_loc;

        __logger_ostream_proxy(source_location loc)
            : m_loc(loc)
        {}

        template <class T>
        __logger_ostream_proxy &operator<<(T const &x) {
            if constexpr (std::is_same_v<std::decay_t<T>, char *>) {
                if (x == endl) {
                    return *this;
                }
            }
            ss << x;
            return *this;
        }

        ~__logger_ostream_proxy() {
            if (ss.str().size())
                log_debug({"{}", m_loc}, ss.str());
        }
    };

    source_location m_loc;

    __logger_ostream(source_location loc = source_location::current())
        : m_loc(loc)
    {}

    auto operator()(source_location loc = source_location::current()) const {
        return __logger_ostream(loc);
    }

    template <class T>
    __logger_ostream_proxy &operator<<(T const &x) {
        return __logger_ostream_proxy(m_loc) << x;
    }
} cout, cerr, clog;

template <class ...Ts>
void log_printf(__with_source_location<const char *> fmt, Ts &&...ts) {
    auto s = cformat(fmt, std::forward<Ts>(ts)...);
    if (s.size() && s[s.size() - 1] == '\n')
        s.resize(s.size() - 1);
    log_debug({"{}", fmt.location()}, s);
}

}

}
