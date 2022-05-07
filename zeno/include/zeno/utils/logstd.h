#pragma once

#include <zeno/utils/log.h>

namespace zeno {

namespace __logstd {

static inline constexpr char endl[] = "\n";

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
    auto s = cformat(fmt.value(), std::forward<Ts>(ts)...);
    if (s.size() && s[s.size() - 1] == '\n')
        s.resize(s.size() - 1);
    log_debug({"{}", fmt.location()}, s);
}

}

}
