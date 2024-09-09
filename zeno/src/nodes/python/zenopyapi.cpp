#ifdef ZENO_WITH_PYTHON
#define PY_SSIZE_T_CLEAN
#include "zenopyapi.h"

//get graph by path
static PyObject*
zeno_getGraph(PyObject* self, PyObject* args)
{
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    std::string graphPath(path);
    auto mainGraph = zeno::getSession().mainGraph;
    if (!mainGraph)
    {
        PyErr_SetString(PyExc_Exception, "Current main graph is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    std::shared_ptr<zeno::Graph> spGraph = mainGraph->getGraphByPath(graphPath);
    if (!spGraph)
    {
        //PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' is invalid").arg(graphPath).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    PyObject* argList = Py_BuildValue("s", graphPath.c_str());
    PyObject* result = PyObject_CallFunctionObjArgs((PyObject*)&SubgraphType, argList, NULL);
    Py_DECREF(argList);
    return result;
}

static PyObject*
zeno_createSubnet(PyObject* self, PyObject* args)
{
    const char* graph_path, *name;
    int type = 0;

    if (!PyArg_ParseTuple(args, "s|s|i", &graph_path, &name, &type))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    std::string graphPath(graph_path);
    std::string subnetName(name);

    auto mainGraph = zeno::getSession().mainGraph;
    if (!mainGraph)
    {
        PyErr_SetString(PyExc_Exception, "Current main graph is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    std::shared_ptr<zeno::Graph> spGraph = mainGraph->getGraphByPath(graphPath);
    if (!spGraph)
    {
        //PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' is invalid").arg(graphPath).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    std::shared_ptr<zeno::INode> spSubnetNode = spGraph->createNode("Subnet", name);

    PyObject* argList = Py_BuildValue("s", subnetName.c_str());
    PyObject* result = PyObject_CallFunctionObjArgs((PyObject*)&SubgraphType, argList, NULL);
    Py_DECREF(argList);
    return result;
}

//module functions.
static PyMethodDef ZenoMethods[] = {
    {"graph", zeno_getGraph, METH_VARARGS, "Get the existing graph by path"},
    {"createSubnet", zeno_createSubnet, METH_VARARGS, "Create a subnet by given graph"},
    {NULL, NULL, 0, NULL}
};



static PyModuleDef zenomodule = {
    PyModuleDef_HEAD_INIT,
    "zeno",
    "Example module that creates an extension type.",
    -1,
    ZenoMethods
};
#endif

PyMODINIT_FUNC
PyInit_zeno(void)
{
    if (PyType_Ready(&SubgraphType) < 0)
        return NULL;

    if (PyType_Ready(&ZNodeType) < 0)
        return NULL;

    PyObject* m = PyModule_Create(&zenomodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&SubgraphType);
    if (PyModule_AddObject(m, "Graph", (PyObject*)&SubgraphType) < 0) {
        Py_DECREF(&SubgraphType);
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "Node", (PyObject*)&ZNodeType) < 0) {
        Py_DECREF(&ZNodeType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
