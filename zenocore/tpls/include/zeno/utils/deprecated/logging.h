#pragma once

#include <sstream>

#if ZLOG_USE_ANDROID
#include <android/log.h>
#endif

namespace zinc::logging {
    enum class LogLevel : char {
        trace = 't',
        debug = 'd',
        info = 'i',
        critical = 'c',
        warning = 'w',
        error = 'e',
        fatal = 'f',
    };

#if ZLOG_USE_ANDROID
    static inline void log_print(LogLevel level, const char *msg) {
        __android_log_print(level == LogLevel::trace ? ANDROID_LOG_VERBOSE :
            level == LogLevel::debug ? ANDROID_LOG_DEBUG :
            level == LogLevel::info ? ANDROID_LOG_INFO :
            level == LogLevel::critical ? ANDROID_LOG_WARN :
            level == LogLevel::warning ? ANDROID_LOG_WARN :
            level == LogLevel::error ? ANDROID_LOG_ERROR :
            level == LogLevel::fatal ? ANDROID_LOG_FATAL :
            ANDROID_LOG_INFO, "zlog", "%s", msg);
    }
#else
    static inline void log_print(LogLevel level, const char *msg) {
        fprintf(stderr, "zlog/%c: %s\n", (char)level, msg);
    }
#endif

    template <class Os>
    void _impl_format(Os &os, const char *fmt) {
        os << fmt;
    }

    template <class Os, class T, class ...Ts>
    void _impl_format(Os &os, const char *fmt, T const &t, Ts &&...ts) {
        const char *p = fmt;
        for (; *p; p++) {
            if (*p == '{' && p[1] == '}') {
                os << t;
                _impl_format(os, p + 2, std::forward<Ts>(ts)...);
                return;
            }
            os << *p;
        }
    }

    template <class Os, class ...Ts>
    void format(Os &os, const char *fmt, Ts &&...ts) {
        _impl_format(os, std::string(fmt).c_str(), std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void log(LogLevel level, const char *fmt, Ts &&...ts) {
        std::stringstream ss;
        format(ss, fmt, std::forward<Ts>(ts)...);
        log_print(level, ss.str().c_str());
    }

    template <class ...Ts>
    void trace(Ts &&...ts) {
        log(LogLevel::trace, std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void debug(Ts &&...ts) {
        log(LogLevel::debug, std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void info(Ts &&...ts) {
        log(LogLevel::info, std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void critical(Ts &&...ts) {
        log(LogLevel::critical, std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void warning(Ts &&...ts) {
        log(LogLevel::warning, std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void error(Ts &&...ts) {
        log(LogLevel::error, std::forward<Ts>(ts)...);
    }

    template <class ...Ts>
    void fatal(Ts &&...ts) {
        log(LogLevel::fatal, std::forward<Ts>(ts)...);
    }
};
