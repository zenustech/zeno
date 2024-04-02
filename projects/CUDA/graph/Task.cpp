#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

namespace detail {
void python_evaluate(const std::string& fmtStr, const std::vector<const char*> args) {
// #if defined(ZENO_WITH_PYTHON3) or ZS_PYTHON_FOUND
    std::string evalStmt = "print(\'" + fmtStr + "\'";
    if (args.size()) {
      evalStmt += ").format("+std::string(args[0]);
      for (int i = 1; i != args.size(); ++i)
          evalStmt += ", " + std::string(args[i]);
      evalStmt += ")";
    }
    evalStmt += ")";

    PyRun_SimpleString(
        "import sys\n"
        "from io import StringIO\n"
        "old_stdout = sys.stdout\n"  // Backup the original stdout
        "sys.stdout = StringIO()\n"  // Replace stdout with a StringIO object to capture outputs
    );
    PyRun_SimpleString(evalStmt.data());
    fmt::print("assembled cmd str: {}\n", evalStmt);

    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *stdOut = PyObject_GetAttrString(sys, "stdout");
    PyObject *output = PyObject_GetAttrString(stdOut, "getvalue");
    PyObject *result = PyObject_CallObject(output, NULL);

    // Convert the captured output to a C++ string
    const char* result_cstr = PyUnicode_AsUTF8(result);
    std::string result_str(result_cstr);

    // Print the captured output in C++
    fmt::print("Captured Python output: {}\n", result_str);

    // Restore the original stdout
    PyRun_SimpleString("sys.stdout = old_stdout");

    // Finalize the Python interpreter
    Py_Finalize();
}
}

struct CommandGenerator : INode {
  void apply() override {
    auto tag = get_input2<std::string>("name_tag");
    if (tag.empty())
      throw std::runtime_error("work name must not be empty!");

    auto cmdFmtStr = get_input2<std::string>("cmd_fmt_string");

    auto args = get_input<ListObject>("arguments");
    std::vector<const char*> as;
    for (auto &&arg : args->get())
      if (auto ptr = dynamic_cast<StringObject *>(arg.get()); ptr != nullptr)
        as.push_back(ptr->get().c_str());
    // auto cmd = fmt::format(cmdFmtStr, as);
    // fmt::print("parsed: [[{}]]\n", cmd);
    detail::python_evaluate(cmdFmtStr, as);

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