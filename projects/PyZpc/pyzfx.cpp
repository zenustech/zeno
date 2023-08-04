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
                /* pFunc is a new reference */
                {
                    for (int i = 0; i < 2; ++i) {
                        pValue = {py_long_c, args[i]};
                        /* pValue reference stolen here: */
                        PyTuple_SetItem(pArgs, i, pValue); //pyobj{py_long_c, args[i]});
                    }
                    // pass a string as the 3rd param
                    pValue = {py_string_c, "|test_string|"};
                    PyTuple_SetItem(pArgs, 2, pValue);
                    // pass a ptr as the 4th param
                    pValue = {py_long_c, vs.data()};
                    PyTuple_SetItem(pArgs, 3, pValue);
                    // PyList, PyDict

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