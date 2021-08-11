#pragma once

#include <zeno/utils/defs.h>
#include <string_view>
#include <string>

namespace zeno {

class BaseException : public std::exception {
private:
  std::string msg;
public:
  ZENO_API BaseException(std::string_view msg) noexcept;
  ZENO_API ~BaseException() noexcept;
  ZENO_API char const *what() const noexcept;
};

class Exception : public BaseException {
public:
  ZENO_API Exception(std::string_view msg) noexcept;
};

}
