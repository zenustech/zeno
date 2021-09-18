#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<zany> arr;

  template <class T = std::shared_ptr<IObject>>
  auto get() {
      std::vector<T> res;
      for (auto const &val: arr) {
          res.push_back(safe_any_cast<T>(val));
      }
      return res;
  }
};

}
