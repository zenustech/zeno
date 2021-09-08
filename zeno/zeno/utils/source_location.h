#pragma once

#if __has_include(<experimental/source_location>)
#include <experimental/source_location>
namespace zeno {
using std::experimental::source_location;
};
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
