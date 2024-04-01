#include "zensim/ZpcTuple.hpp"
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
    std::variant<int, zs::tuple<int, int, int>, std::vector<int>>;

struct WorkNode : IObject {
  int localIndex;

  std::map<std::string, GenericWorkAttribute> attributes;
  std::map<std::string, std::shared_ptr<WorkNode>> deps;
};

struct CommandGenerator : INode {
  void apply() override {
    auto tag = get_input2<std::string>("name_tag");
    if (tag.empty())
      throw std::runtime_error("work name must not be empty!");

    auto cmdFmtStr = get_input2<std::string>("cmd_fmt_string");
    auto cmd = fmt::format(cmdFmtStr, 0);

    // fmt::print("parsed: [[{}]]\n", cmd);
    auto ret = std::make_shared<WorkNode>();

    set_output("job", ret);
  }
};

ZENO_DEFNODE(CommandGenerator)
({/* inputs: */
  {
      {"string", "name_tag", ""},
      {"count"}, // int, int list, int range
      {"int", "batch_size", "1"},

      {"string", "cmd_fmt_string", ""},
      {"list", "arguments"},
      {"DictObject", "options"},

      {"DictObject", "attributes"},
  },
  /* outputs: */
  {
      {"JobNode", "job"},
  },
  /* params: */
  {},
  /* category: */
  {
      "task",
  }});

} // namespace zeno