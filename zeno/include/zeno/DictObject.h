#pragma once

#include <zeno/zeno.h>
#include <vector>
#include <memory>

namespace zeno {

struct DictObject : IObjectClone<DictObject> {
  std::map<std::string, std::shared_ptr<IObject>> lut;
};

}
