#pragma once

#include <exception>
#include <string>

namespace ztd {

class Exception : public std::exception {
    std::string msg;
public:
    Exception(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() const noexcept { return msg.c_str(); }
    ~Exception() noexcept = default;
};

}
