#pragma once

#include <zeno/common.h>
#include <string>
#include <cassert>
#include <exception>

#define RAPIDJSON_NAMESPACE ZENO_NAMESPACE::rapidjson

namespace RAPIDJSON_NAMESPACE {

class AssertionFailure : public std::exception {
    std::string msg;
public:
    AssertionFailure(std::string &&msg) noexcept : msg(std::move(msg)) {}
    virtual const char *what() const noexcept { return msg.c_str(); }
    ~AssertionFailure() noexcept = default;
};

}

#define RAPIDJSON_NOEXCEPT_ASSERT(x) assert(x)
#define RAPIDJSON_ASSERT(x) do { \
    if (!(x)) { \
        throw rapidjson::AssertionFailure("RAPIDJSON_ASSERT(" #x ")"); \
    } \
} while (0)
