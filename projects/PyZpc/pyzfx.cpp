#include "pywrapper.hpp"
//
#include "Structures.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/io/Filesystem.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include <zeno_PyZpc_config.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/utils/log.h>
#include <zeno/types/GenericObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/utils/string.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/CAPIInternals.h>
#include <cwchar>
#include <utility>
#include <thread>
#include <cstdlib>

namespace zeno {
namespace {
static int subprogram_python_main(int argc, char **argv) {
    return Py_BytesMain(argc, argv);
}

static std::wstring s2ws(std::string const &s) {
    std::wstring ws(s.size(), L' '); // Overestimate number of code points.
    ws.resize(std::mbstowcs(ws.data(), s.data(), s.size())); // Shrink to fit.
    return ws;
}

using callback_t = std::function<void(std::optional<std::any>)>; 
static callback_t zpc_init_callback = [] (auto _) {
    auto exe_dir = zs::abs_exe_directory(); 
    Py_SetPythonHome(s2ws(exe_dir).c_str()); 
    log_debug("Initializing Python...");
    Py_Initialize(); 
#ifdef _WIN32
    exe_dir = replace_all(exe_dir, "\\", "/");
#endif
    auto zeno_lib_path = exe_dir + "/" + ZENO_PYZPC_DLL_FILE; 
    auto py_libs_dir = exe_dir + "/resource/py_libs"; 
    if (PyRun_SimpleString(("import os; os.environ['PYTHONPATH'] = '" + exe_dir + "/DLLs';").c_str()) < 0) {
        log_warn("Failed to initialize Python module");
        return;
    }
    if (PyRun_SimpleString(("os.environ['PYTHONHOME'] = '" + exe_dir + "';").c_str()) < 0) {
        log_warn("Failed to initialize Python module");
        return;
    }
#if 0
    if (PyRun_SimpleString(("import sys; import os; sys.path.append(os.path.join('" +
        exe_dir + "', 'DLLs')); ").c_str()) < 0) {
        log_warn("Failed to initialize Python module");
        return;
    }
#endif
    if (PyRun_SimpleString(("import sys; sys.path.append('" + 
        py_libs_dir + "'); import zpy; zpy.init_zeno_lib('" + zeno_lib_path + 
        "'); zpy.zeno_lib_path = '" + zeno_lib_path + "'").c_str()) < 0) {
        log_warn("Failed to initialize Python module");
        return;
    }
    log_debug("Initialized Python successfully!");
    getSession().userData().set("subprogram_python", std::make_shared<GenericObject<int(*)(int, char **)>>(subprogram_python_main));
}; 
static int defPyZpcInit = getSession().eventCallbacks->hookEvent("init", zpc_init_callback);
static callback_t zpc_exit_callback = [] (auto _) {
    Py_Finalize();
}; 
static int defPyZpcExit = getSession().eventCallbacks->hookEvent("exit", zpc_exit_callback);
}

// ref: PythonScript node by Archibate
struct PyZfx : INode {
    void apply() override {
#if 0 
        std::vector<int> vs{1, 2, 3, 10, 20, 30, -10, -20, -30};
        Py_Initialize();
        //
        // ref: https://www.cnblogs.com/panliu/p/4485183.html
        //
        PyRun_SimpleString("import sys");
        auto pstr = fmt::format("sys.path.append(\'{}\')", zs::abs_exe_directory() + "/" + "resource/");
        for (char &c : pstr)
            if (c == '\\')
                c = '/';
        PyRun_SimpleString(pstr.data());
        // PyRun_SimpleString("print(\'Hello World\')");

        fmt::print("checking appended sys path: {}\n", pstr);
        //
        // ref: https://docs.python.org/3/extending/embedding.html#pure-embedding
        //
        {
            // pName = PyUnicode_DecodeFSDefault("HelloWorld");
            pyobj pName = PyUnicode_DecodeFSDefault("HelloWorld");
            pyobj pModule = {py_module_c, pName};
            pyobj pValue;
            pyobj pArgs{py_tuple_c, 4};

            fmt::print("done import\n");

            long args[2] = {33, 2};
            if (pModule) {
                pyobj pFunc{py_func_c, pModule, "multiply"};
                {
                    for (int i = 0; i < 2; ++i)
                        pArgs.setItem(i, pyobj{py_long_c, args[i]});
                    // pass a string as the 3rd param
                    pArgs.setItem(2, pyobj{py_string_c, "|test_string|"});
                    // pass a ptr as the 4th param
                    pArgs.setItem(3, pyobj(py_long_c, vs.data()));
                    // PyList, PyDict
                    // pValue = {py_dict_c};

                    pValue = PyObject_CallObject(pFunc, pArgs);
                    if (pValue) {
                        printf("Result of call: %ld\n", (long)pValue);
                    } else {
                        PyErr_Print();
                        fprintf(stderr, "Call failed\n");
                        exit(1);
                    }
                }
            }
        }

        Py_Finalize();
        fmt::print(fg(fmt::color::blue), "done pyzfx node test.\n");
#endif 
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
        PyObject *zenoMod = PyImport_AddModule("zpy");
        PyObject *zenoModDict = PyModule_GetDict(zenoMod);
        if (PyDict_SetItemString(zenoModDict, "_rets", retsDict) < 0)
            throw makeError("failed to set zpy._rets");
        if (PyDict_SetItemString(zenoModDict, "_args", argsDict) < 0)
            throw makeError("failed to set zpy._args");
 
        if (path.empty()) {
            auto code = get_input2<std::string>("code");
            mainMod = PyRun_StringFlags(code.c_str(), Py_file_input, globals, globals, NULL);
        } else {
            FILE *fp = fopen(path.c_str(), "r");
            if (!fp) {
                perror(path.c_str());
                throw makeError("cannot open file for read: " + path);
            } else {
                if (PyRun_SimpleString(("__import__('sys').path.insert(0, __import__('os').path.dirname('" + path + "'));").c_str()) < 0) {
                    log_warn("Failed to initialize Python import path");
                    return;
                }
                mainMod = PyRun_FileExFlags(fp, path.c_str(), Py_file_input, globals, globals, 1, NULL);
            }
        }
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
ZENO_DEFNODE(PyZfx)({/* inputs: */ 
    {
        {"string", "code", ""},
        {"readpath", "path", ""},
        {"DictObject", "args"},
    },
    /* outputs: */
    {
        {"DictObject", "rets"},
    },
    /* params: */
    {},
    /* category: */
    {
        "pyzfx",
    }});

} // namespace zeno