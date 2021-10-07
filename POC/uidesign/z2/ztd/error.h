#pragma once

#include <string>
#include <exception>
#include <z2/ztd/format.h>

namespace z2::ztd {

class error : public std::exception {
    std::string msg;
public:
    error(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() const noexcept { return msg.c_str(); }
    ~error() noexcept = default;
};

template <class ...Ts>
inline auto make_error(Ts const &...ts) {
    return error(to_string<Ts...>(ts...));
}

template <class ...Ts>
inline auto format_error(const char *fmt, Ts &&...ts) {
    return error(format<Ts...>(fmt, std::forward<Ts>(ts)...));
}

template <class F>
void catch_error(F const &f) {
    try {
        f();
    } catch (std::exception const &e) {
        println("exception occurred:\n", e.what());
    }
}

}
