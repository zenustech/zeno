#pragma once

#include <zeno/zeno.h>
#include <vector>
#include <memory>

namespace zeno {

struct ListObject : IObjectClone<ListObject> {
  std::vector<std::shared_ptr<IObject>> arr;
};

}
