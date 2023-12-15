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
  ZENO_API ~BaseException() noexcept override;
  ZENO_API char const *what() const noexcept override;
#else
  BaseException(std::string_view msg) noexcept : msg(msg) {}
  ~BaseException() noexcept = default;
  char const *what() const noexcept { return msg.c_str(); }
#endif
  BaseException(BaseException const &) = default;
  BaseException &operator=(BaseException const &) = default;
  BaseException(BaseException &&) = default;
  BaseException &operator=(BaseException &&) = default;
};

class Exception : public BaseException {
public:
#ifndef ZENO_APIFREE
  [[deprecated("use makeError(...) in <zeno/utils/Error.h> instead")]]
  ZENO_API Exception(std::string_view msg) noexcept;
#else
  Exception(std::string_view msg) noexcept : BaseException(msg) {}
#endif
};

}
