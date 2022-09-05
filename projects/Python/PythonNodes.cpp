#include <Python.h>
#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/utils/log.h>
#include <zeno/types/GenericObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/CAPIInternals.h>
#include <zeno_Python_config.h>
#include <cwchar>
#include <utility>

namespace zeno {

namespace {

static int subprogram_python_main(int argc, char **argv) {
    return Py_BytesMain(argc, argv);
}

static int defPythonInit = getSession().eventCallbacks->hookEvent("init", [] {
    log_debug("Initializing Python...");
    std::string s = getAssetDir(ZENO_PYTHON_HOME_DIR);
    std::wstring ws(s.size(), L' '); // Overestimate number of code points.
    ws.resize(std::mbstowcs(ws.data(), s.data(), s.size())); // Shrink to fit.
    Py_SetPythonHome(ws.c_str());
    Py_Initialize();
    std::string libpath = getAssetDir(ZENO_PYTHON_LIB_DIR);
    std::string dllfile = ZENO_PYTHON_DLL_FILE;
    if (PyRun_SimpleString(("__import__('sys').path.insert(0, '" + libpath + "'); import ze; ze.initDLLPath('" + dllfile + "')").c_str()) < 0) {
        log_warn("Failed to initialize Python module");
        return;
    }
    log_debug("Initialized Python successfully!");
    getSession().userData().set("subprogram_python", std::make_shared<GenericObject<int(*)(int, char **)>>(subprogram_python_main));
});

static int defPythonExit = getSession().eventCallbacks->hookEvent("exit", [] {
    Py_Finalize();
});

struct PythonFunctor {
    PyObject *pyFunc;

    explicit PythonFunctor(PyObject *pyFunc_) : pyFunc(pyFunc_) {
        Py_INCREF(pyFunc);
    }

    PythonFunctor(PythonFunctor const &that) : pyFunc(that.pyFunc) {
        Py_INCREF(pyFunc);
    }

    PythonFunctor &operator=(PythonFunctor const &that) {
        if (std::addressof(that) != this) {
            Py_DECREF(pyFunc);
            pyFunc = that.pyFunc;
            Py_INCREF(pyFunc);
        }
        return *this;
    }

    ~PythonFunctor() {
        Py_DECREF(pyFunc);
    }

    std::map<std::string, zany> operator()(std::map<std::string, zany> args) const {
        std::map<std::string, zany> rets;
        PyObject *pyKwargs = PyDict_New();
        scope_exit pyKwargsDel = [=] {
            Py_DECREF(pyKwargs);
        };
        for (auto const &[key, val]: args) {
            Zeno_Object handle = capiLoadObjectSharedPtr(val);
            auto valLong = PyLong_FromUnsignedLongLong(handle);
            scope_exit valLongDel = [=] {
                Py_DECREF(valLong);
            };
            if (PyDict_SetItemString(pyKwargs, key.c_str(), valLong) < 0)
                throw makeError("failed to set kwargs item " + key);
        }
        PyObject *pyArgs = PyList_New(0);
        scope_exit pyArgsDel = [=] {
            Py_DECREF(pyArgs);
        };
        PyObject *pyRet = PyObject_Call(pyFunc, pyArgs, pyKwargs);
        scope_exit pyRetDel = [=] {
            Py_DECREF(pyRet);
        };
        if (!PyDict_Check(pyRet)) {
            Zeno_Object handle = PyLong_AsUnsignedLongLong(pyRet);
            if (handle == -1 && PyErr_Occurred()) {
                throw makeError("failed to cast rets value as integer");
            }
            rets.emplace("ret", capiFindObjectSharedPtr(handle));
        } else {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(pyRet, &pos, &key, &value)) {
                Py_ssize_t keyLen = 0;
                const char *keyDat = PyUnicode_AsUTF8AndSize(key, &keyLen);
                if (keyDat == nullptr) {
                    throw makeError("failed to cast rets key as string");
                }
                std::string keyStr(keyDat, keyLen);
                Zeno_Object handle = PyLong_AsUnsignedLongLong(value);
                if (handle == -1 && PyErr_Occurred()) {
                    throw makeError("failed to cast rets value as integer");
                }
                rets.emplace(std::move(keyStr), capiFindObjectSharedPtr(handle));
            }
        }
        return rets;
    }
};

static Zeno_Object factoryFunctionObject(void *inObj_) {
    PyObject *tmpFunc = reinterpret_cast<PyObject *>(inObj_);
    auto funcObj = std::make_shared<FunctionObject>(PythonFunctor(tmpFunc));
    Zeno_Object funcHandle = capiLoadObjectSharedPtr(funcObj);
    return funcHandle;
}

static int defFunctionObjectFactory = capiRegisterObjectFactory("FunctionObject", factoryFunctionObject);

//static Zeno_Object dictObjFactory() {
    //PyObject *zenoMod = PyImport_AddModule("ze");
    //PyObject *zenoModDict = PyModule_GetDict(zenoMod);
    //PyObject *tmpFunc = PyDict_GetItemString(zenoModDict, "_tmp");
    //if (!tmpFunc) throw makeError("cannot access variable ze._tmp");
    //auto funcObj = std::make_shared<FunctionObject>(PythonFunctor(tmpFunc));
    //Zeno_Object funcHandle = capiLoadObjectSharedPtr(funcObj);
    //return funcHandle;
//}

//static int defDictObjFactory = capiRegisterObjectFactory("DictObject", funcObjFactory);

struct PythonScript : INode {
    void apply() override {
        auto args = has_input("args") ? get_input<DictObject>("args") : std::make_shared<DictObject>();
        auto path = get_input2<std::string>("path");
        int ret;
        PyObject *argsDict = PyDict_New();
        scope_exit argsDel = [=] {
            Py_DECREF(argsDict);
        };
        PyObject *retsDict = PyDict_New();
        scope_exit retsDel = [=] {
            Py_DECREF(retsDict);
        };
        std::vector<Zeno_Object> needToDel;
        scope_exit needToDelEraser = [&] {
            for (auto handle: needToDel) {
                capiEraseObjectSharedPtr(handle);
            }
        };
        for (auto const &[k, v]: args->lut) {
            auto handle = capiLoadObjectSharedPtr(v);
            needToDel.push_back(handle);
            PyObject *handleLong = PyLong_FromUnsignedLongLong(handle);
            scope_exit handleDel = [=] {
                Py_DECREF(handleLong);
            };
            if (PyDict_SetItemString(argsDict, k.c_str(), handleLong) < 0) {
                throw makeError("failed to invoke PyDict_SetItemString");
            }
        }
        PyObject *mainMod = PyImport_AddModule("__main__");
        if (!mainMod) throw makeError("failed to get module '__main__'");
        PyObject *globals = PyModule_GetDict(mainMod);
        PyObject *locals = PyDict_New();
        scope_exit localsDel = [=] {
            Py_DECREF(locals);
        };
        PyObject *zenoMod = PyImport_AddModule("ze");
        PyObject *zenoModDict = PyModule_GetDict(zenoMod);
        if (PyDict_SetItemString(zenoModDict, "_rets", retsDict) < 0)
            throw makeError("failed to set ze._rets");
        if (PyDict_SetItemString(zenoModDict, "_args", argsDict) < 0)
            throw makeError("failed to set ze._args");
        std::shared_ptr<Graph> currGraphSP = getThisGraph()->shared_from_this();  // TODO
        Zeno_Graph currGraphHandle = capiLoadGraphSharedPtr(currGraphSP);
        scope_exit currGraphEraser = [=] {
            capiEraseGraphSharedPtr(currGraphHandle);
        };
        {
            PyObject *currGraphLong = PyLong_FromUnsignedLongLong(currGraphHandle);
            scope_exit currGraphLongDel = [=] {
                Py_DECREF(currGraphLong);
            };
            if (PyDict_SetItemString(zenoModDict, "_currgraph", currGraphLong) < 0)
                throw makeError("failed to set ze._currgraph");
        }
        scope_exit currGraphLongReset = [=] {
            PyObject *currGraphLongZero = PyLong_FromUnsignedLongLong(0);
            scope_exit currGraphLongZeroDel = [=] {
                Py_DECREF(currGraphLongZero);
            };
            (void)PyDict_SetItemString(zenoModDict, "_currgraph", currGraphLongZero);
        };
        if (path.empty()) {
            auto code = get_input2<std::string>("code");
            mainMod = PyRun_StringFlags(code.c_str(), Py_file_input, globals, locals, NULL);
        } else {
            FILE *fp = fopen(path.c_str(), "r");
            if (!fp) {
                perror(path.c_str());
                throw makeError("cannot open file for read: " + path);
            } else {
                mainMod = PyRun_FileExFlags(fp, path.c_str(), Py_file_input, globals, locals, 1, NULL);
            }
        }
        currGraphLongReset.reset();
        currGraphEraser.reset();
        needToDelEraser.reset();
        if (!mainMod) {
            PyErr_Print();
            throw makeError("Python exception occurred, see console for more details");
        }
        auto rets = std::make_shared<DictObject>();
        {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(retsDict, &pos, &key, &value)) {
                Py_ssize_t keyLen = 0;
                const char *keyDat = PyUnicode_AsUTF8AndSize(key, &keyLen);
                if (keyDat == nullptr) {
                    throw makeError("failed to cast rets key as string");
                }
                std::string keyStr(keyDat, keyLen);
                Zeno_Object handle = PyLong_AsUnsignedLongLong(value);
                if (handle == -1 && PyErr_Occurred()) {
                    throw makeError("failed to cast rets value as integer");
                }
                rets->lut.emplace(std::move(keyStr), capiFindObjectSharedPtr(handle));
            }
        }
        {
            PyObject *retsRAIIDict = PyDict_GetItemString(zenoModDict, "_retsRAII");
            PyDict_Clear(retsRAIIDict);
        }
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
