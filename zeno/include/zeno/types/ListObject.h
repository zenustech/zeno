#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<zany> arr;

  template <class T>
  auto get() {
      std::vector<T> res;
      for (auto const &val: arr) {
          res.push_back(smart_any_cast<T>(val));
      }
      return res;
  }

#ifndef ZENO_APIFREE
  ZENO_API virtual void dumpfile(std::string const &path) override;
#else
  virtual void dumpfile(std::string const &path) override {}
#endif
};

}
