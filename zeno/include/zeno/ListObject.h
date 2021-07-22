#pragma once

#include <zeno/zeno.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<std::shared_ptr<IObject>> arr;

#ifndef ZEN_NOREFDLL
  ZENAPI virtual void dumpfile(std::string const &path) override;
#else
  virtual void dumpfile(std::string const &path) override {}
#endif
};

}
