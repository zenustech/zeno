#pragma once

#include <zeno/core/IObject.h>
#include <zeno/funcs/LiterialConverter.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<zany> arr;

  std::set<int> dirtyIndice;                        //该list下dirty的obj的index
  std::map<std::string, int> nodeNameArrItemMap;    //obj所在的节点名到obj在arr中索引的map

  ListObject() = default;

  explicit ListObject(std::vector<zany> arrin) : arr(std::move(arrin)) {
  }

  template <class T = IObject>
  std::vector<std::shared_ptr<T>> get() const {
      std::vector<std::shared_ptr<T>> res;
      for (auto const &val: arr) {
          res.push_back(safe_dynamic_cast<T>(val));
      }
      return res;
  }

  template <class T = IObject>
  std::vector<T *> getRaw() const {
      std::vector<T *> res;
      for (auto const &val: arr) {
          res.push_back(safe_dynamic_cast<T>(val.get()));
      }
      return res;
  }

  template <class T>
  std::vector<T> get2() const {
      std::vector<T> res;
      for (auto const &val: arr) {
          res.push_back(objectToLiterial<T>(val));
      }
      return res;
  }

  template <class T>
  [[deprecated("use get2<T>() instead")]]
  std::vector<T> getLiterial() const {
      return get2<T>();
  }
};

}
