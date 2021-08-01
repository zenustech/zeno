#pragma once

#define FMT_HEADER_ONLY

#include <iostream>
#include <string_view>

namespace zlog {
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
    void format(Os &os, std::string_view const &fmt, Ts &&...ts) {
        _impl_format(os, std::string(fmt).c_str(), std::forward<Ts>(ts)...);
    }

    template <class Os, class ...Ts>
    void print(std::string_view const &fmt, Ts &&...ts) {
        format(std::cout, fmt, std::forward<Ts>(ts)...);
    }

    enum class LogLevel : char {
        trace = 't',
        debug = 'd',
        info = 'i',
        critical = 'c',
        warning = 'w',
        error = 'e',
        fatal = 'f',
    };

    template <class Os>
    void _prefix_bar(Os &os, LogLevel level) {
        os << "zlog/" << (char)level << ": ";
    }

    template <class ...Ts>
    void log(LogLevel level, std::string_view const &fmt, Ts &&...ts) {
        _prefix_bar(std::cerr, level);
        format(std::cerr, fmt, std::forward<Ts>(ts)...);
        std::cerr << std::endl;
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
