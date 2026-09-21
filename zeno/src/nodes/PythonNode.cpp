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
#include <zeno/types/PythonContextObject.h>
#include <cwchar>
#include <fstream>
#include <sstream>
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
                {"string", "script"},
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
            {{"string", "nameList"},{"string", "keyWords"},{"readpath", "materialPath"},{"string", "matchInputs"}, {"string", "script"}},
            {},
            {},
            {"command"},
            });

        struct SequentialPythonNode : zeno::INode {
            virtual void apply() override {
                auto code = get_input2<std::string>("code");

                // 若提供了 script（python 脚本文件路径），读取其内容作为要执行的
                // python 代码，与 code 输入走完全相同的流程（链命名空间、zeArgs
                // 注入、编译执行）。code 为空时二者效果一致。
                if (has_input("script")) {
                    auto scriptPath = get_input2<std::string>("script");
                    if (!scriptPath.empty()) {
                        std::ifstream fs(scriptPath, std::ios::binary);
                        if (!fs.is_open())
                            throw makeError("cannot open python script: " + scriptPath);
                        std::ostringstream ss;
                        ss << fs.rdbuf();
                        code = ss.str();
                    }
                }

                auto prevContext = has_input("prev_context") ? get_input<PythonContextObject>("prev_context") : nullptr;
                auto args = has_input("zeArgs") ? get_input<DictObject>("zeArgs") : nullptr;

                // 下方所有 Python C-API 调用都需要当前线程持有 GIL。
                // 由于调度器可能在工作线程上执行本节点，故整个函数体
                // 包裹在 GIL 的获取/释放之间，保证调用与共享 dict 安全。
                PyGILState_STATE gstate = PyGILState_Ensure();
                zeno::scope_exit gstateRelease = [=] { PyGILState_Release(gstate); };

                PyObject* globals = nullptr;
                bool ownsGlobals = false;
                // 仅当上游上下文确实持有可用 globals 时才沿用；否则按链根新建。
                bool reusePrevContext = prevContext && prevContext->getGlobals() != nullptr;
                if (reusePrevContext) {
                    // 沿用上游节点建立的链内命名空间。该引用为借用引用，
                    // 由输入持有，我们不得拥有或释放。
                    globals = prevContext->getGlobals();
                } else {
                    // 链根：新建私有命名空间，不复用进程级 __main__ dict，
                    // 避免独立链相互泄漏变量。此 dict 为自有的新引用，
                    // 交由上下文对象持有后再释放本处引用。
                    globals = PyDict_New();
                    if (!globals)
                        throw makeError("failed to create python globals dict");
                    PyObject* builtins = PyEval_GetBuiltins();
                    if (!builtins)
                        throw makeError("failed to get python builtins");
                    PyDict_SetItemString(globals, "__builtins__", builtins);
                    ownsGlobals = true;
                }
                // 链根新建的 globals 即使中途异常也必须释放，避免错误路径泄漏
                // 引用（正常路径由下方新建的上下文对象持有其自身引用）。
                zeno::scope_exit globalsRelease = [=] {
                    if (ownsGlobals)
                        Py_DECREF(globals);
                };

                // 参数只存在于链私有 globals 中，不落进程级单例。以保留名
                // `zeArgs` 绑定到该链命名空间，脚本用 `zeArgs["key"]` 取值，
                // 既不与普通变量冲突，也不触碰全局 `ze` 模块。
                if (args) {
                    PyObject* argsDict = PyDict_New();
                    if (!argsDict)
                        throw makeError("failed to create python args dict");
                    // 中途类型报错等异常路径也要释放 argsDict，避免引用泄漏。
                    zeno::scope_exit argsDictRelease = [=] {
                        Py_DECREF(argsDict);
                    };
                    for (auto const& [k, v] : args->lut) {
                        PyObject* pyVal = nullptr;
                        if (auto numObj = std::dynamic_pointer_cast<NumericObject>(v)) {
                            auto& val = numObj->get();
                            std::visit([&](auto const& e) {
                                using T = std::decay_t<decltype(e)>;
                                if constexpr (std::is_same_v<T, bool>) {
                                    pyVal = PyBool_FromLong(e ? 1 : 0);
                                } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
                                    pyVal = PyLong_FromLongLong(static_cast<long long>(e));
                                } else if constexpr (std::is_integral_v<T> && !std::is_signed_v<T>) {
                                    pyVal = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(e));
                                } else if constexpr (std::is_same_v<T, float>) {
                                    pyVal = PyFloat_FromDouble(e);
                                } else if constexpr (std::is_same_v<T, double>) {
                                    pyVal = PyFloat_FromDouble(e);
                                }
                            }, val);
                        } else if (auto strObj = std::dynamic_pointer_cast<StringObject>(v)) {
                            pyVal = PyUnicode_FromString(strObj->get().c_str());
                        }
                        // 其余 zeno 类型目前静默丢弃；此处显式报错，
                        // 让用户知道该参数不可用。
                        if (!pyVal)
                            throw makeError("unsupported python arg type for key: " + k);
                        if (PyDict_SetItemString(argsDict, k.c_str(), pyVal) < 0) {
                            Py_DECREF(pyVal);
                            throw makeError("failed to set python arg: " + k);
                        }
                        Py_DECREF(pyVal);
                    }
                    // 将 `zeArgs` 绑定到链命名空间，供脚本读取参数，
                    // 无需共享任何进程级状态。argsDict 的引用由 argsDictRelease
                    // 在本块结束处释放。
                    PyDict_SetItemString(globals, "zeArgs", argsDict);
                }

                PyObject* compiled = Py_CompileString(code.c_str(), "<code>", Py_file_input);
                if (!compiled) {
                    PyErr_Print();
                    throw makeError("failed to compile code");
                }

                PyObject* result = PyEval_EvalCode(compiled, globals, globals);
                Py_DECREF(compiled);
                if (!result) {
                    PyErr_Print();
                    throw makeError("Python exception occurred, see console for more details");
                }
                Py_DECREF(result);

                if (reusePrevContext) {
                    // 沿链直接传递同一个共享上下文。
                    set_output("context", prevContext);
                } else {
                    // 将 globals 交给上下文对象（其持有自己的引用）；链根
                    // 新建的临时引用由开头注册的 globalsRelease 统一释放，
                    // 异常路径同样被覆盖，避免泄漏或重复 DECREF。
                    auto outputContext = std::make_shared<PythonContextObject>(globals);
                    set_output("context", outputContext);
                }
            }
        };

        ZENDEFNODE(SequentialPythonNode, {
            {
                {"readpath", "script"},
                {"multiline_string", "code"},
                {"dict", "zeArgs"},
                {"pythoncontext", "prev_context"},
            },
            {
                {"pythoncontext", "context"},
            },
            {},
            {"command"},
            });

        // CopyPythonObject：为分叉链路复制一份独立的 python 命名空间副本。
        // 当一个 SequentialPythonNode 的输出被多条链引用时，各自给每条独立链再接
        // 一个本节点，使其在各自 globals 副本中运行，避免变量/zeArgs 互相覆盖。
        // 采用浅拷贝（PyDict_Copy）：新建字典、键空间隔离而值对象仍共享，
        // 开销极小、不会因不可拷贝对象抛异常；各分叉链各自写新变量互不影响。
        struct CopyPythonObject : zeno::INode {
            virtual void apply() override {
                auto src = has_input<PythonContextObject>("context")
                    ? get_input<PythonContextObject>("context") : nullptr;

                PyGILState_STATE gstate = PyGILState_Ensure();
                zeno::scope_exit gstateRelease = [=] { PyGILState_Release(gstate); };

                PyObject* out = nullptr;
                if (src && src->getGlobals()) {
                    out = PyDict_Copy(src->getGlobals());  // 浅拷贝：新 dict 共享值
                    if (!out)
                        PyErr_Print();
                }
                if (!out) {
                    out = PyDict_New();  // 上游无有效 globals 时退化为空命名空间
                    if (!out) {
                        PyErr_Print();
                        throw makeError("failed to create python globals copy");
                    }
                }

                auto outCtx = std::make_shared<PythonContextObject>(out);
                Py_DECREF(out);
                set_output("context", outCtx);
            }
        };

        ZENDEFNODE(CopyPythonObject, {
            {
                {"pythoncontext", "context"},
            },
            {
                {"pythoncontext", "context"},
            },
            {},
            {"command"},
        });
    }
}
#endif