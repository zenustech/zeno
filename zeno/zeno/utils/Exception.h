#pragma once

#include <zeno/utils/api.h>
#include <string_view>
#include <string>

namespace zeno {

class BaseException : public std::exception {
private:
  std::string msg;
public:
#ifndef ZENO_APIFREE
  ZENO_API BaseException(std::string_view msg) noexcept;
  ZENO_API ~BaseException() noexcept;
  ZENO_API char const *what() const noexcept;
#else
  BaseException(std::string_view msg) noexcept : msg(msg) {}
  ~BaseException() noexcept = default;
  char const *what() const noexcept { return msg.c_str(); }
#endif
};

class Exception : public BaseException {
public:
#ifndef ZENO_APIFREE
  ZENO_API Exception(std::string_view msg) noexcept;
#else
  Exception(std::string_view msg) noexcept : BaseException(msg) {}
#endif
};

}
