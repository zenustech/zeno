#include "pywrapper.hpp"
//
#include "Structures.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/resource/Filesystem.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct PyZfx : INode {
    void apply() override {
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
#if 1
        PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
        pName = PyUnicode_DecodeFSDefault("HelloWorld");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);
#else
        PyObject *pArgs, *pValue;
        pyobj pName = PyUnicode_DecodeFSDefault("HelloWorld");
        pyobj pModule = PyImport_Import(pName);
#endif

        fmt::print("done import\n");

        long args[2] = {33, 2};
        if (pModule) {
#if 1
            pFunc = PyObject_GetAttrString(pModule, "multiply");
#else
            pyobj pFunc = PyObject_GetAttrString(pModule, "multiply");
#endif
            /* pFunc is a new reference */

            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(4);
                for (int i = 0; i < 2; ++i) {
                    pValue = PyLong_FromLong(args[i]);
                    if (!pValue) {
                        Py_DECREF(pArgs);
#if 1
                        Py_DECREF(pModule);
#endif
                        fprintf(stderr, "Cannot convert argument\n");
                        exit(1);
                    }
                    /* pValue reference stolen here: */
                    PyTuple_SetItem(pArgs, i, pValue);
                }
                // pass a string as the 3rd param
                pValue = PyUnicode_InternFromString("|test_string|");
                PyTuple_SetItem(pArgs, 2, pValue);
                // pass a ptr as the 4th param
                pValue = PyLong_FromVoidPtr(vs.data());
                PyTuple_SetItem(pArgs, 3, pValue);
                // PyList, PyDict

                pValue = PyObject_CallObject(pFunc, pArgs);
                Py_DECREF(pArgs);
                if (pValue != NULL) {
                    printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                    Py_DECREF(pValue);
                } else {
#if 1
                    Py_DECREF(pFunc);
                    Py_DECREF(pModule);
#endif
                    PyErr_Print();
                    fprintf(stderr, "Call failed\n");
                    exit(1);
                }
            } else {
                if (PyErr_Occurred())
                    PyErr_Print();
                fprintf(stderr, "Cannot find function \"%s\"\n", "multiply");
            }
#if 1
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
#endif
        }

        Py_Finalize();
        fmt::print(fg(fmt::color::blue), "done pyzfx node test.\n");
    }
};
ZENDEFNODE(PyZfx, {/* inputs: */ {},
                   /* outputs: */
                   {},
                   /* params: */
                   {},
                   /* category: */
                   {
                       "pyzfx",
                   }});

} // namespace zeno