#pragma once

#include <string>
#include <exception>
#include <spdlog/spdlog.h>

namespace zs::ztd::ztd {

class error : public std::exception {
    std::string msg;
public:
    error(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() const noexcept { return msg.c_str(); }
    ~error() noexcept = default;
};

template <class ...Args>
inline auto format_error(fmt::format_string<Args...> fmt, Args &&...args) {
    return error(fmt::format(fmt, std::forward<Args>(args)...));
}

}
