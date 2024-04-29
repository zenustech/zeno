#pragma once

#include "zensim/ZpcReflection.hpp"
#include "zensim/ZpcTuple.hpp"
#include "zensim/types/Polymorphism.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <cctype>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/string.h>
#include <zeno/zeno.h>

namespace zeno {

using GenericWorkAttribute =
    std::variant<zs::tuple<int, int, int>, std::vector<int>>;

struct WorkNode : IObject {
  //
  std::string tag;
  std::vector<std::string> workItems;
  std::map<std::string, std::shared_ptr<WorkNode>> deps;
  // reserved
  std::map<std::string, GenericWorkAttribute> attributes;
};

} // namespace zeno