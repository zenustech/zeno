#ifdef ZENO_WITH_PYTHON3
#include <Python.h>
#include <QtWidgets>
#include "zeno/utils/log.h"
#include "zeno/zeno.h"
#include "pythonenv.h"

PyMODINIT_FUNC PyInit_zeno(void);

#ifdef WIN32
#define ZENO_PYTHON_DLL_FILE "zeno.dll"
#else
#define ZENO_PYTHON_DLL_FILE "libzeno.so"
#endif

static std::wstring s2ws(std::string const& s) {
    std::wstring ws(s.size(), L' '); // Overestimate number of code points.
    ws.resize(std::mbstowcs(ws.data(), s.data(), s.size())); // Shrink to fit.
    return ws;
}

void initPythonEnv(const char* progName)
{
    wchar_t* program = Py_DecodeLocale(progName, NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    if (PyImport_AppendInittab("zeno", PyInit_zeno) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    std::wstring homedir = s2ws(std::string(ZENO_PYTHON_HOME));
    Py_SetPythonHome(homedir.c_str());

    Py_SetProgramName(program);

    Py_Initialize();

    PyObject* pmodule = PyImport_ImportModule("zeno");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'zeno'\n");
    }

    std::string libpath(ZENO_PYTHON_MODULE_DIR);
    std::string dllfile = ZENO_PYTHON_DLL_FILE;
    if (PyRun_SimpleString(("__import__('sys').path.insert(0, '" + libpath + "'); import ze; ze.init_zeno_lib('" + dllfile + "')").c_str()) < 0) {
        zeno::log_warn("Failed to initialize Python module");
        return;
    }
    zeno::log_debug("Initialized Python successfully!");

    PyMem_RawFree(program);
}
#endif