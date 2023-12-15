#define PY_SSIZE_T_CLEAN
#include "zenopyapi.h"
#include <QtWidgets>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/enum.h>
#include <zenomodel/include/nodesmgr.h>


static PyObject*
zeno_getGraph(PyObject* self, PyObject* args)
{
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return Py_None;

    QString graphName = QString::fromUtf8(name);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return Py_None;

    QModelIndex idxGraph = pModel->index(graphName);
    std::string _graphName = graphName.toStdString();

    PyObject* argList = Py_BuildValue("s", _graphName.c_str());
    PyObject* result = PyObject_CallFunctionObjArgs((PyObject*)&SubgraphType, argList, NULL);
    Py_DECREF(argList);
    return result;
}

static PyObject*
zeno_createGraph(PyObject* self, PyObject* args)
{
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return Py_None;

    QString graphName = QString::fromUtf8(name);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return Py_None;

    QModelIndex idxGraph = pModel->index(graphName);
    if (idxGraph.isValid())
    {
        //already exists.
        //return Py_None;
    }
    else
    {
        pModel->newSubgraph(graphName);
        idxGraph = pModel->index(graphName);
    }

    std::string _graphName = graphName.toStdString();
    PyObject* argList = Py_BuildValue("s", _graphName.c_str());
    PyObject* result = PyObject_CallFunctionObjArgs((PyObject*)&SubgraphType, argList, NULL);
    Py_DECREF(argList);
    return result;
}

static PyObject*
zeno_removeGraph(PyObject* self, PyObject* args)
{
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return Py_None;

    QString graphName = QString::fromUtf8(name);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return Py_None;

    QModelIndex idxGraph = pModel->index(graphName);
    if (idxGraph.isValid())
    {
        pModel->removeGraph(idxGraph.row());
    }
    return Py_None;
}

static PyObject*
zeno_forkGraph(PyObject* self, PyObject* args)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Py_None;

    char *_subgName, *_ident;
    if (!PyArg_ParseTuple(args, "ss", &_subgName, &_ident))
        return Py_None;

    const QString& subgName = QString::fromUtf8(_subgName);
    const QString& ident = QString::fromUtf8(_ident);

    const QModelIndex& subgIdx = pModel->index(subgName);
    const QModelIndex& subnetnodeIdx = pModel->index(ident, subgIdx);
    const QModelIndex& forknode = pModel->fork(subgIdx, subnetnodeIdx);

    const std::string& forkident = forknode.data(ROLE_OBJID).toString().toStdString();

    PyObject* argList = Py_BuildValue("ss", _subgName, forkident.c_str());

    PyObject* result = PyObject_CallObject((PyObject*)&ZNodeType, argList);
    Py_DECREF(argList);

    return Py_None;
}


//module functions.
static PyMethodDef ZenoMethods[] = {
    {"graph", zeno_getGraph, METH_VARARGS, "Get the existing graph on current scene"},
    {"createGraph", zeno_createGraph, METH_VARARGS, "Create a subgraph"},
    {"removeGraph", zeno_removeGraph, METH_VARARGS, "Remove a subgraph"},
    {"forkGraph", zeno_forkGraph, METH_VARARGS, "Fork a subgraph"},
    {NULL, NULL, 0, NULL}
};



static PyModuleDef zenomodule = {
    PyModuleDef_HEAD_INIT,
    "zeno",
    "Example module that creates an extension type.",
    -1,
    ZenoMethods
};

PyMODINIT_FUNC
PyInit_zeno(void)
{
    PyObject* m;
    if (PyType_Ready(&SubgraphType) < 0)
        return NULL;

    if (PyType_Ready(&ZNodeType) < 0)
        return NULL;

    m = PyModule_Create(&zenomodule);
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