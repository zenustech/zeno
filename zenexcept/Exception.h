#pragma once

#include <string>
#include <cstring>

namespace zpp {

void attachDebugger();

struct Exception : std::exception {
public:
    Exception() noexcept {
        attachDebugger();
    }
};

struct SignalException : Exception {
    int signo;
    SignalException(int signo) noexcept
        : signo(signo) {}

    char const *what() const noexcept {
        return strsignal(signo);
    }
};

void attachDebugger();

}
