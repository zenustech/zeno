#pragma once

#include <zen/zen.h>
#include <variant>
#include <array>

namespace zen {

struct StringObject : IObjectClone<StringObject> {
  std::string value;

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
