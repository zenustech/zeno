#ifdef ZENO_WITH_PYTHON3
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
#include <zeno/utils/string.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/CAPIInternals.h>
#include <cwchar>
#include <utility>
#include <thread>


#ifdef WIN32
#define ZENO_PYTHON_DLL_FILE "zeno.dll"
#else
#define ZENO_PYTHON_DLL_FILE "libzeno.so"
#endif


namespace zeno {
    namespace {

        static int subprogram_python_main(int argc, char** argv) {
            return Py_BytesMain(argc, argv);
        }

        static std::wstring s2ws(std::string const& s) {
            std::wstring ws(s.size(), L' '); // Overestimate number of code points.
            ws.resize(std::mbstowcs(ws.data(), s.data(), s.size())); // Shrink to fit.
            return ws;
        }


        static int defPythonInit = getSession().eventCallbacks->hookEvent("init", [] {
#if 0
            log_debug("Initializing Python...");
            std::wstring homedir = s2ws(std::string(ZENO_PYTHON_HOME));
            Py_SetPythonHome(homedir.c_str());
            static std::string execName = getConfigVariable("EXECFILE");

            Py_SetProgramName(s2ws(execName).c_str());

            Py_Initialize();

            std::string libpath(ZENO_PYTHON_MODULE_DIR);
            std::string dllfile = ZENO_PYTHON_DLL_FILE;
            if (PyRun_SimpleString(("__import__('sys').path.insert(0, '" + libpath + "'); import ze; ze.init_zeno_lib('" + dllfile + "')").c_str()) < 0) {
                log_warn("Failed to initialize Python module");
                return;
            }
            log_debug("Initialized Python successfully!");

            //getSession().userData().set("subprogram_python", std::make_shared<GenericObject<int(*)(int, char**)>>(subprogram_python_main));
#endif
            });

        static int defPythonExit = getSession().eventCallbacks->hookEvent("exit", [] {
            Py_Finalize();
            });

        struct PythonFunctor {
            PyObject* pyFunc;

            explicit PythonFunctor(PyObject* pyFunc_) : pyFunc(pyFunc_) {
                Py_INCREF(pyFunc);
            }

            PythonFunctor(PythonFunctor const& that) : pyFunc(that.pyFunc) {
                Py_INCREF(pyFunc);
            }

            PythonFunctor& operator=(PythonFunctor const& that) {
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
                PyObject* pyKwargs = PyDict_New();
                scope_exit pyKwargsDel = [=] {
                    Py_DECREF(pyKwargs);
                };
                for (auto const& [key, val] : args) {
                    Zeno_Object handle = capiLoadObjectSharedPtr(val);
                    auto valLong = PyLong_FromUnsignedLongLong(handle);
                    scope_exit valLongDel = [=] {
                        Py_DECREF(valLong);
                    };
                    if (PyDict_SetItemString(pyKwargs, key.c_str(), valLong) < 0)
                        throw makeError("failed to set kwargs item " + key);
                }
                PyObject* pyArgs = PyList_New(0);
                scope_exit pyArgsDel = [=] {
                    Py_DECREF(pyArgs);
                };
                PyObject* pyRet = PyObject_Call(pyFunc, pyArgs, pyKwargs);
                if (!pyRet) {
                    PyErr_Print();
                    throw makeError("Python exception occurred (during function call), see console for more details");
                }
                scope_exit pyRetDel = [=] {
                    Py_DECREF(pyRet);
                };
                scope_exit pyFuncRetsRAIIDel = [=] {
                    if (PyObject_HasAttrString(pyFunc, "_wrapRetRAII"))
                        PyObject_DelAttrString(pyFunc, "_wrapRetRAII");
                };
                if (PyDict_Check(pyRet)) {
                    PyObject* key, * value;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(pyRet, &pos, &key, &value)) {
                        Py_ssize_t keyLen = 0;
                        const char* keyDat = PyUnicode_AsUTF8AndSize(key, &keyLen);
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
                else if (pyRet != Py_None) {
                    Zeno_Object handle = PyLong_AsUnsignedLongLong(pyRet);
                    if (handle == -1 && PyErr_Occurred()) {
                        throw makeError("failed to cast rets value as integer");
                    }
                    rets.emplace("ret", capiFindObjectSharedPtr(handle));
                }
                return rets;
            }
        };

        static Zeno_Object factoryFunctionObject(void* inObj_) {
            PyObject* tmpFunc = reinterpret_cast<PyObject*>(inObj_);
            auto funcObj = std::make_shared<FunctionObject>(PythonFunctor(tmpFunc));
            Zeno_Object funcHandle = capiLoadObjectSharedPtr(funcObj);
            return funcHandle;
        }

        static int defFunctionObjectFactory = capiRegisterObjectFactory("FunctionObject", factoryFunctionObject);

        static PyObject* callFunctionObjectCFunc(PyObject* pyHandleAndKwargs_) {
            PyObject* pyHandleVal = PyTuple_GetItem(pyHandleAndKwargs_, 0);
            PyObject* pyKwargs = PyTuple_GetItem(pyHandleAndKwargs_, 1);
            Zeno_Object obj = PyLong_AsUnsignedLongLong(pyHandleVal);
            auto* objFunc = safe_dynamic_cast<FunctionObject>(capiFindObjectSharedPtr(obj).get(), "callFunctionObjectCFunc");
            FunctionObject::DictType objParams;
            {
                PyObject* key, * value;
                Py_ssize_t pos = 0;
                if (!PyDict_Check(pyKwargs)) throw makeError("expect to pyArgs_ be an dict");
                while (PyDict_Next(pyKwargs, &pos, &key, &value)) {
                    Py_ssize_t keyLen = 0;
                    const char* keyDat = PyUnicode_AsUTF8AndSize(key, &keyLen);
                    if (keyDat == nullptr) {
                        throw makeError("failed to cast rets key as string");
                    }
                    std::string keyStr(keyDat, keyLen);
                    Zeno_Object handle = PyLong_AsUnsignedLongLong(value);
                    if (handle == -1 && PyErr_Occurred()) {
                        throw makeError("failed to cast rets value as integer");
                    }
                    objParams.emplace(std::move(keyStr), capiFindObjectSharedPtr(handle));
                }
            }
            objParams = objFunc->call(objParams);
            PyDict_Clear(pyKwargs);
            for (auto const& [k, v] : objParams) {
                PyObject* handleObj = PyLong_FromUnsignedLongLong(capiLoadObjectSharedPtr(v));
                PyDict_SetItemString(pyKwargs, k.c_str(), handleObj);
            }
            return pyKwargs;
        }

        static int defCallFunctionObjectCFunc = capiRegisterCFunctionPtr("FunctionObject_call", reinterpret_cast<void* (*)(void*)>(callFunctionObjectCFunc));

        static void* defactoryFunctionObject(Zeno_Object inHandle_) {
            auto objSp = capiFindObjectSharedPtr(inHandle_);
            auto funcObj = dynamic_cast<FunctionObject*>(objSp.get());
            if (!funcObj) throw makeError<TypeError>(typeid(FunctionObject), typeid(*objSp),
                "convert from zeno function to python function");
            //PyObject *selfPtr = PyDict_New();
            PyObject* pyHandleVal = PyLong_FromUnsignedLongLong(inHandle_);
            if (!pyHandleVal) throw makeError("failed to invoke PyLong_FromUnsignedLongLong");
            return reinterpret_cast<void*>(pyHandleVal);
        }

        static int defFunctionObjectDefactory = capiRegisterObjectDefactory("FunctionObject", defactoryFunctionObject);

        struct PythonNode : zeno::INode {
            void apply() override {
                bool onlyui_gen = get_param<bool>("onlyui");
                if (onlyui_gen)
                    return;

                auto args = has_input("args") ? get_input<DictObject>("args") : std::make_shared<DictObject>();
                auto path = has_input("path") ? get_input2<std::string>("path") : "";
                int ret;
                PyObject* argsDict = PyDict_New();
                scope_exit argsDel = [=] {
                    Py_DECREF(argsDict);
                };
                PyObject* retsDict = PyDict_New();
                scope_exit retsDel = [=] {
                    Py_DECREF(retsDict);
                };
                std::vector<Zeno_Object> needToDel;
                scope_exit needToDelEraser = [&] {
                    for (auto handle : needToDel) {
                        capiEraseObjectSharedPtr(handle);
                    }
                };
                for (auto const& [k, v] : args->lut) {
                    auto handle = capiLoadObjectSharedPtr(v);
                    needToDel.push_back(handle);
                    PyObject* handleLong = PyLong_FromUnsignedLongLong(handle);
                    scope_exit handleDel = [=] {
                        Py_DECREF(handleLong);
                    };
                    if (PyDict_SetItemString(argsDict, k.c_str(), handleLong) < 0) {
                        throw makeError("failed to invoke PyDict_SetItemString");
                    }
                }
                PyObject* mainMod = PyImport_AddModule("__main__");
                if (!mainMod) throw makeError("failed to get module '__main__'");
                PyObject* globals = PyModule_GetDict(mainMod);
                PyObject* zenoMod = PyImport_AddModule("ze.zeno");
                PyObject* zenoModDict = PyModule_GetDict(zenoMod);
                if (PyDict_SetItemString(zenoModDict, "_rets", retsDict) < 0)
                    throw makeError("failed to set ze._rets");
                if (PyDict_SetItemString(zenoModDict, "_args", argsDict) < 0)
                    throw makeError("failed to set ze._args");
                std::shared_ptr<Graph> currGraphSP = getThisGraph()->shared_from_this();  // TODO
                Zeno_Graph currGraphHandle = capiLoadGraphSharedPtr(currGraphSP);
                //scope_exit currGraphEraser = [=] {
                    //capiEraseGraphSharedPtr(currGraphHandle);
                //};
                {
                    PyObject* currGraphLong = PyLong_FromUnsignedLongLong(currGraphHandle);
                    scope_exit currGraphLongDel = [=] {
                        Py_DECREF(currGraphLong);
                    };
                    if (PyDict_SetItemString(zenoModDict, "_currgraph", currGraphLong) < 0)
                        throw makeError("failed to set ze._currgraph");
                }
                //scope_exit currGraphLongReset = [=] {
                    //PyObject *currGraphLongZero = PyLong_FromUnsignedLongLong(0);
                    //scope_exit currGraphLongZeroDel = [=] {
                        //Py_DECREF(currGraphLongZero);
                    //};
                    //(void)PyDict_SetItemString(zenoModDict, "_currgraph", currGraphLongZero);
                //};
                if (path.empty()) {
                    auto code = get_input2<std::string>("script");
                    mainMod = PyRun_StringFlags(code.c_str(), Py_file_input, globals, globals, NULL);
                    PyErr_Print();
                }
                else {
                    FILE* fp = fopen(path.c_str(), "r");
                    if (!fp) {
                        perror(path.c_str());
                        throw makeError("cannot open file for read: " + path);
                    }
                    else {
                        mainMod = PyRun_FileExFlags(fp, path.c_str(), Py_file_input, globals, globals, 1, NULL);
                    }
                }
                //currGraphLongReset.reset();
                //currGraphEraser.reset();
                needToDelEraser.reset();
                //if (!mainMod) {
                //    PyErr_Print();
                //    throw makeError("Python exception occurred, see console for more details");
                //}
                auto rets = std::make_shared<DictObject>();
                {
                    PyObject* key, * value;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(retsDict, &pos, &key, &value)) {
                        Py_ssize_t keyLen = 0;
                        const char* keyDat = PyUnicode_AsUTF8AndSize(key, &keyLen);
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
                    PyObject* retsRAIIDict = PyDict_GetItemString(zenoModDict, "_retsRAII");
                    if (retsRAIIDict)
                        PyDict_Clear(retsRAIIDict);
                }
                set_output("rets", std::move(rets));
            }
        };

        ZENDEFNODE(PythonNode, {
            {
                {"string", "script", "", Socket_Primitve, CodeEditor},
                {"readpath", "path"},
                {"dict", "args"}
            },
            {
                {"dict", "rets"},
            },
            {
                {"bool", "onlyui", "false"}
            },
            {"command"},
        });

        struct GenerateCommands : zeno::INode {
            virtual void apply() override {

            }
        };

        ZENDEFNODE(GenerateCommands, {
            {
                {"string", "source"},
                {"string", "commands"},
            },
            {},
            {},
            {"command"},
        });

        struct PythonMaterialNode : zeno::INode {
            virtual void apply() override {

            }
        };

        ZENDEFNODE(PythonMaterialNode, {
            {
                {"string", "nameList"},
                {"string", "keyWords"},
                {"string", "materialPath", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
                {"string", "matchInputs"}, 
                {"string", "script", "", Socket_Primitve, Multiline}
            },
            {},
            {},
            {"command"},
            });
    }
}
#endif