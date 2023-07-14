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
        // PyRun_SimpleString("print(\'Hello World\')");

        //
        // ref: https://docs.python.org/3/extending/embedding.html#pure-embedding
        //
        PyObject *pName, *pModule, *pFunc;
#if RESOURCE_AT_RELATIVE_PATH
        auto scriptPath = zs::abs_exe_directory() + "/" + "resource/HelloWorld.py";
#else
        auto scriptPath = std::string{AssetDirPath} + "/" + "Scripts/HelloWorld.py";
#endif
        pName = PyUnicode_DecodeFSDefault(scriptPath.c_str());
        pModule = PyImport_Import(pName);

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