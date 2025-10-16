#ifdef ZENO_WITH_PYTHON
#include <Python.h>
#include <pybind11/pybind11.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include "pythonenv.h"

#ifdef WIN32
#include <Windows.h>
#endif

namespace py = pybind11;

PyMODINIT_FUNC PyInit_zen(void);

void initPythonEnv(const char* progName)
{
    wchar_t* program = Py_DecodeLocale(progName, NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    if (PyImport_AppendInittab("zen", PyInit_zen) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    Py_SetProgramName(program);

    Py_Initialize();

    std::string tempCode;
    tempCode = "import zen";
    if (PyRun_SimpleString(tempCode.c_str()) < 0) {
        zeno::log_warn("Failed to initialize Python module");
        return;
    }

    PyMem_RawFree(program);
}

struct _SGlobal_initPythonEnv
{
    _SGlobal_initPythonEnv() {
#ifdef WIN32
        char filename[MAX_PATH];
        DWORD size = GetModuleFileNameA(NULL, filename, MAX_PATH);
        initPythonEnv(filename);
#else
        initPythonEnv("");
#endif
    }
};

//static _SGlobal_initPythonEnv _inst;


#endif