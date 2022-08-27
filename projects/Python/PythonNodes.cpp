#include <Python.h>
#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>

namespace zeno {
namespace {

struct PythonScript : INode {
    void apply() override {
        auto code = get_input2<std::string>("code");
        auto args = has_input("args") ? get_input<DictObject>("args") : std::make_shared<DictObject>();
        Py_Initialize();
        PyRun_SimpleString(code.c_str());
        Py_Finalize();
        auto rets = std::make_shared<DictObject>();
        set_output("rets", std::move(rets));
    }
};
ZENO_DEFNODE(PythonScript)({
    {
        {"string", "code"},
        {"DictObject", "args"},
    },
    {
        {"DictObject", "rets"},
    },
    {},
    {"python"},
});

}
}
