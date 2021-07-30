#pragma once

#include <zeno/utils/defs.h>
#include <string>

namespace zeno {

class Exception : public std::exception {
private:
  std::string msg;
public:
  ZENO_API Exception(std::string const &msg) noexcept;
  ZENO_API ~Exception() noexcept;
  ZENO_API char const *what() const noexcept;
};

}
