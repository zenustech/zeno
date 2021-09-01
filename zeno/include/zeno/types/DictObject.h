#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/any.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct DictObject : IObjectClone<DictObject> {
  std::map<std::string, zany> lut;

  template <class T>
  auto get() {
      std::map<std::string, T> res;
      for (auto const &[key, val]: lut) {
          res.emplace(key, smart_any_cast<T>(val));
      }
      return res;
  }
};

}
