#pragma once

#include <zeno/core/IObject.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct DictObject : IObjectClone<DictObject> {
  std::map<std::string, zany> lut;

  template <class T = IObject>
  std::map<std::string, std::shared_ptr<T>> get() const {
      std::map<std::string, std::shared_ptr<T>> res;
      for (auto const &[key, val]: lut) {
          res.emplace(key, safe_dynamic_cast<T>(val));
      }
      return res;
  }

  template <class T>
  std::map<std::string, T> getLiterial() const {
      std::map<std::string, T> res;
      for (auto const &[key, val]: lut) {
          res.emplace(key, objectToLiterial<T>(val));
      }
      return res;
  }
};

}
