// ref: https://docs.python.org/3/c-api/intro.html#include-files
#define PY_SSIZE_T_CLEAN
#include <Python.h>
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
        Py_Initialize();
        //
        // ref: https://www.cnblogs.com/panliu/p/4485183.html
        //
        PyRun_SimpleString("import sys");
        PyRun_SimpleString(fmt::format("sys.path.append(\'{}\')", zs::abs_exe_directory() + "/" + "resource/").data());
        // PyRun_SimpleString("print(\'Hello World\')");

        //
        // ref: https://docs.python.org/3/extending/embedding.html#pure-embedding
        //
        PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
        pName = PyUnicode_DecodeFSDefault("HelloWorld");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

	fmt::print("done import\n");

        long args[2] = {33, 2};
        if (pModule != NULL) {
            pFunc = PyObject_GetAttrString(pModule, "multiply");
            /* pFunc is a new reference */

            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(2);
                for (int i = 0; i < 2; ++i) {
                    pValue = PyLong_FromLong(args[i]);
                    if (!pValue) {
                        Py_DECREF(pArgs);
                        Py_DECREF(pModule);
                        fprintf(stderr, "Cannot convert argument\n");
                        exit(1);
                    }
                    /* pValue reference stolen here: */
                    PyTuple_SetItem(pArgs, i, pValue);
                }
                pValue = PyObject_CallObject(pFunc, pArgs);
                Py_DECREF(pArgs);
                if (pValue != NULL) {
                    printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                    Py_DECREF(pValue);
                } else {
                    Py_DECREF(pFunc);
                    Py_DECREF(pModule);
                    PyErr_Print();
                    fprintf(stderr, "Call failed\n");
                    exit(1);
                }
            } else {
                if (PyErr_Occurred())
                    PyErr_Print();
                fprintf(stderr, "Cannot find function \"%s\"\n", "multiply");
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
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