#include <Python.h>
#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/utils/log.h>
#include <zeno_Python_config.h>

namespace zeno {
namespace {

static int defPythonInit = getSession().eventCallbacks->hookEvent("init", [] {
    log_debug("Initializing Python...");
    Py_Initialize();
    log_debug("Initialized Python successfully!");
});

static int defPythonExit = getSession().eventCallbacks->hookEvent("exit", [] {
    log_debug("Finalizing Python...");
    Py_Finalize();
    log_debug("Finalized Python successfully!");
});

struct PythonScript : INode {
    void apply() override {
        auto code = get_input2<std::string>("code");
        auto args = has_input("args") ? get_input<DictObject>("args") : std::make_shared<DictObject>();
        std::string libpath = getAssetDir(ZENO_PYTHON_LIB_DIR);
        std::string dllfile = ZENO_PYTHON_DLL_FILE;
        PyRun_SimpleString(("__import__('sys').path.insert(0, '" + libpath + "'); import zeno; zeno._zenodllfile = '" + dllfile + "'").c_str());
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
