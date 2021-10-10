#pragma once

#include <string>
#include <exception>

#define RAPIDJSON_NAMESPACE rapidjson

namespace RAPIDJSON_NAMESPACE {

class AssertionFailure : public std::exception {
    std::string msg;
public:
    AssertionFailure(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() const noexcept { return msg.c_str(); }
    ~AssertionFailure() noexcept = default;
};

}

#define RAPIDJSON_ASSERT(x) do { \
    if (!(x)) { \
        throw rapidjson::AssertionFailure(#x); \
    } \
} while (0)
