#pragma once

#include "../TypeAlias.hpp"
#include "zensim/execution/Intrinsics.hpp"

namespace zs {

  /// implementation copied from c++10/experimental/source_location
  // ref:
  // https://github.com/microsoft/STL/blob/62137922ab168f8e23ec1a95c946821e24bde230/stl/inc/source_location
  // https://github.com/microsoft/STL/issues/54
  struct source_location {
    // 14.1.2, source_location creation
    static constexpr source_location current(const char* __file = __builtin_FILE(),
                                             const char* __func = __builtin_FUNCTION(),
                                             int __line = __builtin_LINE(),
                                             int __col = 0) noexcept {
      source_location __loc;
      __loc._M_file = __file;
      __loc._M_func = __func;
      __loc._M_line = __line;
      __loc._M_col = __col;
      return __loc;
    }

    constexpr source_location() noexcept
        : _M_file("unknown"), _M_func(_M_file), _M_line(0), _M_col(0) {}

    // 14.1.3, source_location field access
    constexpr u32 line() const noexcept { return _M_line; }
    constexpr u32 column() const noexcept { return _M_col; }
    constexpr const char* file_name() const noexcept { return _M_file; }
    constexpr const char* function_name() const noexcept { return _M_func; }

  private:
    const char* _M_file;
    const char* _M_func;
    u32 _M_line;
    u32 _M_col;
  };

}  // namespace zs