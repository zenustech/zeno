#include <Python.h>
#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/utils/log.h>
#include <zeno/core/CAPI.h>
#include <zeno_Python_config.h>

namespace zeno {

ZENO_API Zeno_Object capiLoadObjectSharedPtr(std::shared_ptr<IObject> const &objPtr_);
ZENO_API Zeno_Object capiEraseObjectSharedPtr(Zeno_Object object_);

namespace {

static int defPythonInit = getSession().eventCallbacks->hookEvent("init", [] {
    log_debug("Initializing Python...");
    Py_Initialize();
    std::string libpath = getAssetDir(ZENO_PYTHON_LIB_DIR);
    std::string dllfile = ZENO_PYTHON_DLL_FILE;
    PyRun_SimpleString(("__import__('sys').path.insert(0, '" + libpath + "'); import ze; ze.initDLLPath('" + dllfile + "')").c_str());
    log_debug("Initialized Python successfully!");
});

static int defPythonExit = getSession().eventCallbacks->hookEvent("exit", [] {
    log_debug("Finalizing Python...");
    Py_Finalize();
    log_debug("Finalized Python successfully!");
});

struct PythonScript : INode {
    void apply() override {
        auto args = has_input("args") ? get_input<DictObject>("args") : std::make_shared<DictObject>();
        auto path = get_input2<std::string>("path");
        int ret;
        if (path.empty()) {
            auto code = get_input2<std::string>("code");
            PyObject *argsDict = PyDict_New();
            std::vector<Zeno_Object> needToDel;
            for (auto const &[k, v]: args->lut) {
                auto h = capiLoadObjectSharedPtr(v);
                needToDel.push_back(h);
                PyDict_SetItemString(argsDict, k.c_str(), PyLong_FromUnsignedLongLong(h));
            }
            for (auto *p: needToDel) {
                capiEraseObjectSharedPtr(p);
            }
            ret = PyRun_String(code.c_str());
        } else {
            FILE *fp = fopen(path.c_str(), "r");
            if (!fp) {
                perror(path.c_str());
                throw makeError("cannot open file for read: " + path);
            } else {
                ret = PyRun_SimpleFile(fp, path.c_str());  // will do fclose for us
            }
        }
        if (ret != 0) {
            PyErr_Print();
            throw makeError("Python exception occurred, see log for more details");
        }
        auto rets = std::make_shared<DictObject>();
        set_output("rets", std::move(rets));
    }
};
ZENO_DEFNODE(PythonScript)({
    {
        {"string", "code"},
        {"readpath", "path"},
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
