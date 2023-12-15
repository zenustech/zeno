#pragma once

#if __has_include(<experimental/source_location>)
#include <experimental/source_location>
namespace zeno {
using std::experimental::source_location;
};
#else

#if defined(_MSC_VER)

#if __has_include(<yvals_core.h>)
#include <yvals_core.h>
#endif
#include <cstdint>

namespace zeno {
    // copied from: https://github.com/microsoft/STL/blob/main/stl/inc/source_location
    struct source_location {
    static constexpr source_location current(const uint_least32_t _Line_ = __builtin_LINE(),
        const uint_least32_t _Column_ = __builtin_COLUMN(), const char* const _File_ = __builtin_FILE(),
        const char* const _Function_ = __builtin_FUNCTION()) noexcept {
        source_location _Result{};
        _Result._Line     = _Line_;
        _Result._Column   = _Column_;
        _Result._File     = _File_;
        _Result._Function = _Function_;
        return _Result;
    }

    constexpr source_location() noexcept = default;

    constexpr uint_least32_t line() const noexcept {
        return _Line;
    }
    constexpr uint_least32_t column() const noexcept {
        return _Column;
    }
    constexpr const char* file_name() const noexcept {
        return _File;
    }
    constexpr const char* function_name() const noexcept {
        return _Function;
    }

private:
    uint_least32_t _Line{};
    uint_least32_t _Column{};
    const char* _File     = "";
    const char* _Function = "";
};
}

#else

namespace zeno {
struct source_location
  {

    // 14.1.2, source_location creation
    static constexpr source_location
    current(const char* __file = "unknown",
	    const char* __func = "unknown",
	    int __line = 0,
	    int __col = 0) noexcept
    {
      source_location __loc;
      __loc._M_file = __file;
      __loc._M_func = __func;
      __loc._M_line = __line;
      __loc._M_col = __col;
      return __loc;
    }

    constexpr source_location() noexcept
    : _M_file("unknown"), _M_func(_M_file), _M_line(0), _M_col(0)
    { }

    // 14.1.3, source_location field access
    constexpr unsigned int line() const noexcept { return _M_line; }
    constexpr unsigned int column() const noexcept { return _M_col; }
    constexpr const char* file_name() const noexcept { return _M_file; }
    constexpr const char* function_name() const noexcept { return _M_func; }

  private:
    const char* _M_file;
    const char* _M_func;
    unsigned int _M_line;
    unsigned int _M_col;
  };
}

#endif

#endif
