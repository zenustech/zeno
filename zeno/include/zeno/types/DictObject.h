#pragma once

#include <zeno/zeno.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct DictObject : IObjectClone<DictObject> {
  std::map<std::string, std::shared_ptr<IObject>> lut;
};

}
