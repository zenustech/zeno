#include "zenopyapi.h"
#include <QtWidgets>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/enum.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/uihelper.h>


//init function
static int
Graph_init(ZSubGraphObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwList[] = { "hGraph", NULL };
    ZENO_HANDLE hGraph;
    char* _subgName;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwList, &_subgName))
        return -1;

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return 0;

    self->subgIdx = pModel->index(QString::fromUtf8(_subgName));
    return 0;
}

static PyObject*
Graph_name(ZSubGraphObject* self, PyObject* Py_UNUSED(ignored))
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return Py_None;

    const QString& name = self->subgIdx.data(ROLE_OBJNAME).toString();
    return PyUnicode_FromFormat(name.toUtf8());
}

static PyObject*
Graph_createNode(ZSubGraphObject* self, PyObject* arg)
{
    //todo: support keys
    static char* kwList[] = { "pos", NULL };

    ZENO_HANDLE hGraph;
    //if (!PyArg_ParseTupleAndKeywords(arg, kwds, "i", kwList, &hGraph))
    //    return -1;
    PyObject* _arg = PyTuple_GET_ITEM(arg, 0);
    if (!PyUnicode_Check(_arg)) {
        return Py_None;
    }

    char* nodeCls = nullptr;
    if (!PyArg_Parse(_arg, "s", &nodeCls))
        return Py_None;

    const QString& subgName = self->subgIdx.data(ROLE_OBJNAME).toString();
    const QString& descName = QString::fromUtf8(nodeCls);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    const QString& ident = NodesMgr::createNewNode(pModel, self->subgIdx, descName, QPointF(0, 0));

    //const QModelIndex& nodeIdx = pModel->index(ident, self->subgIdx);
    std::string _subgName = subgName.toStdString();
    std::string _ident = ident.toStdString();
    PyObject* argList = Py_BuildValue("ss", _subgName.c_str(), _ident.c_str());

    PyObject* result = PyObject_CallObject((PyObject*)&ZNodeType, argList);
    Py_DECREF(argList);
    return result;
}

static PyObject*
Graph_getNode(ZSubGraphObject* self, PyObject* arg)
{
    ZENO_HANDLE hGraph;
    //if (!PyArg_ParseTupleAndKeywords(arg, kwds, "i", kwList, &hGraph))
    //    return -1;
    PyObject* _arg = PyTuple_GET_ITEM(arg, 0);
    if (!PyUnicode_Check(_arg)) {
        return Py_None;
    }

    char* _ident = nullptr;
    if (!PyArg_Parse(_arg, "s", &_ident))
        return Py_None;

    const QString& subgName = self->subgIdx.data(ROLE_OBJNAME).toString();
    std::string _subgName = subgName.toStdString();
    PyObject* argList = Py_BuildValue("ss", _subgName.c_str(), _ident);
    PyObject* result = PyObject_CallObject((PyObject*)&ZNodeType, argList);
    Py_DECREF(argList);
    return result;
}

static PyObject*
Graph_addLink(ZSubGraphObject* self, PyObject* arg)
{
    char *_outNode, *_outSock, *_inNode, *_inSock;
    if (!PyArg_ParseTuple(arg, "ssss", &_outNode, &_outSock, &_inNode, &_inSock))
        return Py_None;

    const QString& graphName = self->subgIdx.data(ROLE_OBJNAME).toString();
    const QString& outNode = QString::fromUtf8(_outNode);
    const QString& outSock = QString::fromUtf8(_outSock);
    const QString& inNode = QString::fromUtf8(_inNode);
    const QString& inSock = QString::fromUtf8(_inSock);

    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    EdgeInfo edge;
    edge.inSockPath = UiHelper::constructObjPath(graphName, inNode, "[node]/inputs/", inSock);
    edge.outSockPath = UiHelper::constructObjPath(graphName, outNode, "[node]/outputs/", outSock);
    pModel->addLink(self->subgIdx, edge);
    return Py_None;
}

static PyMethodDef GraphMethods[] = {
    {"name", (PyCFunction)Graph_name, METH_NOARGS, "Return the name of graph"},
    {"createNode", (PyCFunction)Graph_createNode, METH_VARARGS, "Add the node to this graph"},
    {"node", (PyCFunction)Graph_getNode, METH_VARARGS, "Get the node from the graph"},
    {"addLink", (PyCFunction)Graph_addLink, METH_VARARGS, "Add link"},
    {NULL, NULL, 0, NULL}
};

PyTypeObject SubgraphType = {
    // clang-format off
        PyVarObject_HEAD_INIT(nullptr, 0)
        // clang-format on
        "zeno.Graph",                      /* tp_name */
        sizeof(ZSubGraphObject),                /* tp_basicsize */
        0,                                  /* tp_itemsize */
        nullptr,                            /* tp_dealloc */
    #if PY_VERSION_HEX < 0x03080000
        nullptr,                            /* tp_print */
    #else
        0, /* tp_vectorcall_offset */
    #endif
        nullptr,                            /* tp_getattr */
        nullptr,                            /* tp_setattr */
        nullptr,                            /* tp_compare */
        nullptr,                            /* tp_repr */
        nullptr,                            /* tp_as_number */
        nullptr,                            /* tp_as_sequence */
        nullptr,                            /* tp_as_mapping */
        nullptr,                            /* tp_hash */
        nullptr,                            /* tp_call */
        nullptr,                            /* tp_str */
        nullptr,                            /* tp_getattro */
        nullptr,                            /* tp_setattro */
        nullptr,                            /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,                 /* tp_flags */
        PyDoc_STR("Zeno Subgraph objects"), /* tp_doc */
        nullptr,                            /* tp_traverse */
        nullptr,                            /* tp_clear */
        nullptr,                            /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        nullptr,                            /* tp_iter */
        nullptr,                            /* tp_iternext */
        GraphMethods,                            /* tp_methods */
        nullptr,                            /* tp_members */
        nullptr,                            /* tp_getset */
        nullptr,                            /* tp_base */
        nullptr,                            /* tp_dict */
        nullptr,                            /* tp_descr_get */
        nullptr,                            /* tp_descr_set */
        0,                          /* tp_dictoffset */
       (initproc)Graph_init,                /* tp_init */
        nullptr,                            /* tp_alloc */
        PyType_GenericNew,                  /* tp_new */
};