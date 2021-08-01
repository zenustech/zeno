#pragma once

#include <fmt/core.h>
#include <iostream>
#include <string_view>

namespace zlog {
    enum class LogLevel : char {
        trace = 't',
        debug = 'd',
        info = 'i',
        critical = 'c',
        warning = 'w',
        error = 'e',
        fatal = 'f',
    };

    template <class ...Ts>
    void log(LogLevel level, std::string_view const &fmt, Ts &&...ts) {
        std::cout << "zlog/" << (char)level << ": ";
        std::cout << fmt::format(fmt, std::forward<Ts>(ts)...);
        std::cout << std::endl;
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
