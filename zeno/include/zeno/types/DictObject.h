#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/any.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct DictObject : IObjectClone<DictObject> {
  std::map<std::string, any> lut;
};

}
