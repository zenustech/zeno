#pragma once

#include <exception>
#include <string>

namespace zeno::ztd {

class error : public std::exception {
    std::string msg;
public:
    error(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() const noexcept { return msg.c_str(); }
    ~error() noexcept = default;
};

}
