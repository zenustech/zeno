#pragma once

#include <string>
#include <exception>
#include <spdlog/spdlog.h>

namespace zeno2::ztd {

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

//template <class F>
//void catch_error(F const &f) {
    //try {
        //f();
    //} catch (std::exception const &e) {
        //SPDLOG_ERROR("exception occurred:\n{}\n", e.what());
    //}
//}

#define ZENO2_ZTD_ASSERT(x, ...) do { \
    [[unlikely]] if (!(x)) throw ztd::format_error("AssertionError: " #x __VA_OPT__(": ") __VA_ARGS__); \
} while (0)

}
