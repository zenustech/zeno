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
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QString graphName = QString::fromUtf8(name);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QModelIndex idxGraph = pModel->index(graphName);
    if (!idxGraph.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' is invalid").arg(graphName).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
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
    int type = SUBGRAPH_NOR;
 
    if (!PyArg_ParseTuple(args, "s|i", &name, &type))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QString graphName = QString::fromUtf8(name);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QModelIndex idxGraph = pModel->index(graphName);
    if (idxGraph.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' already exist").arg(graphName).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    else
    {
        pModel->newSubgraph(graphName, (SUBGRAPH_TYPE)type);
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
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QString graphName = QString::fromUtf8(name);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QModelIndex idxGraph = pModel->index(graphName);
    if (idxGraph.isValid())
    {
        pModel->removeGraph(idxGraph.row());
    }
    else
    {
        PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' is invalid").arg(graphName).toUtf8());
        PyErr_WriteUnraisable(Py_None);
    }
    return Py_None;
}

static PyObject*
zeno_forkGraph(PyObject* self, PyObject* args)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    char *_subgName, *_ident;
    if (!PyArg_ParseTuple(args, "ss", &_subgName, &_ident))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QString& subgName = QString::fromUtf8(_subgName);
    const QString& ident = QString::fromUtf8(_ident);

    const QModelIndex& subgIdx = pModel->index(subgName);
    if (!subgIdx.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' is inValid").arg(subgName).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    const QModelIndex& subnetnodeIdx = pModel->index(ident, subgIdx);
    if (!subnetnodeIdx.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Node '%1' is inValid").arg(ident).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    const QModelIndex& forknode = pModel->fork(subgIdx, subnetnodeIdx);
    if (!forknode.isValid())
    {
        PyErr_SetString(PyExc_Exception, "Fork failed");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const std::string& forkident = forknode.data(ROLE_OBJID).toString().toStdString();

    PyObject* argList = Py_BuildValue("ss", _subgName, forkident.c_str());

    PyObject* result = PyObject_CallObject((PyObject*)&ZNodeType, argList);
    Py_DECREF(argList);

    return result;
}

static PyObject*
zeno_forkMaterial(PyObject* self, PyObject* args)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    char* _forkName, *_subgName, * _mtlid;
    if (!PyArg_ParseTuple(args, "sss", &_forkName, &_subgName, &_mtlid))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QString& forkName = QString::fromUtf8(_forkName);
    const QString& subgName = QString::fromUtf8(_subgName);
    const QString& mtlid = QString::fromUtf8(_mtlid);

    const QModelIndex& forkIdx = pModel->index(forkName);
    if (!forkIdx.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Subgraph '%1' is inValid").arg(subgName).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QModelIndex& forknode = pModel->forkMaterial(forkIdx, subgName, mtlid, mtlid);
    if (!forknode.isValid())
    {
        PyErr_SetString(PyExc_Exception, "Fork failed");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const std::string& forkident = forknode.data(ROLE_OBJID).toString().toStdString();
    const QModelIndex& sugIdx = forknode.data(ROLE_SUBGRAPH_IDX).toModelIndex();
    const std::string& name = sugIdx.data(ROLE_OBJNAME).toString().toStdString();

    PyObject* argList = Py_BuildValue("ss", name.c_str(), forkident.c_str());

    PyObject* result = PyObject_CallObject((PyObject*)&ZNodeType, argList);
    Py_DECREF(argList);

    return result;
}

static PyObject*
zeno_renameGraph(PyObject* self, PyObject* args)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    char* _subgName, * _newName;
    if (!PyArg_ParseTuple(args, "ss", &_subgName, &_newName))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    const QString& subgName = QString::fromUtf8(_subgName);
    const QString& newName = QString::fromUtf8(_newName);
    pModel->renameSubGraph(subgName, newName);
    return Py_None;
}

//module functions.
static PyMethodDef ZenoMethods[] = {
    {"graph", zeno_getGraph, METH_VARARGS, "Get the existing graph on current scene"},
    {"createGraph", zeno_createGraph, METH_VARARGS, "Create a subgraph"},
    {"removeGraph", zeno_removeGraph, METH_VARARGS, "Remove a subgraph"},
    {"forkGraph", zeno_forkGraph, METH_VARARGS, "Fork a subgraph"},
    {"forkMaterial", zeno_forkMaterial, METH_VARARGS, "Fork a Material subgraph"},
    {"renameGraph", zeno_renameGraph, METH_VARARGS, "Rename a subgraph"},
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