#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<std::shared_ptr<IObject>> arr;

#ifndef ZENO_APIFREE
  ZENO_API virtual void dumpfile(std::string const &path) override;
#else
  virtual void dumpfile(std::string const &path) override {}
#endif
};

}
