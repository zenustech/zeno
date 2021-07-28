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

    static constexpr source_location current(
            const char* __file = __builtin_FILE(),
            const char* __func = __builtin_FUNCTION(),
            int __line = __builtin_LINE(),
            int __col = 0) noexcept {
      source_location __loc{};
      __loc._M_file = __file;
      __loc._M_func = __func;
      __loc._M_line = __line;
      __loc._M_col = __col;
      return __loc;
    }
};

}
