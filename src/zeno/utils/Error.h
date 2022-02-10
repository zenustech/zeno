#pragma once

#include <zeno/utils/Exception.h>
#include <typeinfo>

namespace zeno {

class TypeError : public Exception {
private:
  std::type_info const &expect;
  std::type_info const &got;
  std::string hint;
public:
  ZENO_API TypeError(std::type_info const &expect, std::type_info const &got, std::string const &hint = "nohint") noexcept;
};


class KeyError : public Exception {
private:
  std::string key;
  std::string type;
  std::string hint;
public:
  ZENO_API KeyError(std::string const &key, std::string const &type = "key", std::string const &hint = "nohint") noexcept;
};

}
