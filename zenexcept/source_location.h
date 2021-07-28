#pragma once

namespace zpp {

struct source_location {
    const char *_M_file;
    const char *_M_func;
    int _M_line;
    int _M_col;

    constexpr const char *file_name() const noexcept {
        return _M_file;
    }

    constexpr const char *function_name() const noexcept {
        return _M_func;
    }

    constexpr int line() const noexcept {
        return _M_line;
    }

    constexpr int column() const noexcept {
        return _M_col;
    }

    source_location(
            const char *file = __builtin_FILE(),
            const char *func = __builtin_FUNCTION(),
            int line = __builtin_LINE(),
            int col = 0) noexcept
            : _M_file(file)
            , _M_func(func)
            , _M_line(line)
            , _M_col(col)
    {}
};

#define ZPP_SOURCE_LOCATION zpp::source_location const &location = {}

struct traceback {
    traceback const *_M_prev;
    source_location _M_location;

    traceback(ZPP_SOURCE_LOCATION) : _M_prev(nullptr), _M_location(location) {}

    traceback(traceback const &prev, ZPP_SOURCE_LOCATION) noexcept
        : _M_prev(&prev), _M_location(location) {
    }

    source_location const &location() const noexcept { return _M_location; }
    traceback const *previous() const noexcept { return _M_prev; }
};

#define ZPP_TRACEBACK zpp::traceback zpp_tb = {}

}
