#pragma once

#include <exception>

namespace ztd {

struct Exception : std::exception {
    std::string msg;

    Exception(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() noexcept { return msg.c_str(); }
    ~Exception() noexcept = default;
};

}
