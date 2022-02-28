#pragma once

#include <zeno/core/IObject.h>
#include <string>

namespace zeno {

struct StringObject : IObjectClone<StringObject> {
  std::string value;

  StringObject() = default;
  StringObject(std::string const &value) : value(value) {}

  std::string const &get() const {
    return value;
  }

  std::string &get() {
    return value;
  }

  void set(std::string const &x) {
    value = x;
  }
};

}
