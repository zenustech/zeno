#pragma once

#include <zeno/core/IObject.h>
#include <zeno/funcs/LiterialConverter.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<zany> arr;

  ListObject() = default;

  explicit ListObject(std::vector<zany> arrin) : arr(std::move(arrin)) {
  }

  template <class T = IObject>
  std::vector<std::shared_ptr<T>> get() {
      std::vector<std::shared_ptr<T>> res;
      for (auto const &val: arr) {
          res.push_back(safe_dynamic_cast<T>(val));
      }
      return res;
  }

  template <class T = IObject>
  std::vector<T *> getRaw() {
      std::vector<T *> res;
      for (auto const &val: arr) {
          res.push_back(safe_dynamic_cast<T>(val.get()));
      }
      return res;
  }

  template <class T>
  std::vector<T> getLiterial() {
      std::vector<T> res;
      for (auto const &val: arr) {
          res.push_back(objectToLiterial<T>(val));
      }
      return res;
  }
};

}
