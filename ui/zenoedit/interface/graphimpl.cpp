#ifdef ZENO_WITH_PYTHON3
#include "zenopyapi.h"
#include <QtWidgets>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/enum.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/command.h>
#include "variantptr.h"

static QVariant parseValue(PyObject* v, const QString& type)
{
    QVariant val;
    if (type == "string")
    {
        char* _val = nullptr;
        if (PyArg_Parse(v, "s", &_val))
        {
            val = _val;
        }
    }
    else if (type == "int")
    {
        int _val;
        if (PyArg_Parse(v, "i", &_val))
        {
            val = _val;
        }
    }
    else if (type == "float")
    {
        float _val;
        if (PyArg_Parse(v, "f", &_val))
        {
            val = _val;
        }
    }
    else if (type.startsWith("vec"))
    {
        PyObject* obj;
        if (PyArg_Parse(v, "O", &obj))
        {
            int count = Py_SIZE(obj);
            UI_VECTYPE vec;
            vec.resize(count);
            bool bError = false;
            if (type.contains("i"))
            {
                for (int i = 0; i < count; i++)
                {
                    PyObject* item = PyTuple_GET_ITEM(obj, i);
                    int iVal;
                    if (PyArg_Parse(item, "i", &iVal))
                    {
                        vec[i] = iVal;
                    }
                    else {
                        bError = true;
                        break;
                    }
                }
            }
            else if (type.contains("f"))
            {
                for (int i = 0; i < count; i++)
                {
                    PyObject* item = PyTuple_GET_ITEM(obj, i);
                    float dbVval;
                    if (PyArg_Parse(item, "f", &dbVval))
                    {
                        vec[i] = dbVval;
                    }
                    else {
                        bError = true;
                        break;
                    }
                }
            }
            else
            {
                bError = true;
            }
            if (bError)
            {
                PyErr_SetString(PyExc_Exception, "args error");
                PyErr_WriteUnraisable(Py_None);
                return val;
            }
            val = QVariant::fromValue(vec);
        }
    }
    if (!val.isValid())
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return val;
    }
    return val;
}

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
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return 0;
    }

    self->subgIdx = pModel->index(QString::fromUtf8(_subgName));
    return 0;
}

static PyObject*
Graph_name(ZSubGraphObject* self, PyObject* Py_UNUSED(ignored))
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QString& name = self->subgIdx.data(ROLE_OBJNAME).toString();
    return PyUnicode_FromFormat(name.toUtf8());
}

static PyObject*
Graph_createNode(ZSubGraphObject* self, PyObject* arg, PyObject* kw)
{
    PyObject* _arg = PyTuple_GET_ITEM(arg, 0);
    if (!PyUnicode_Check(_arg)) {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    char* nodeCls = nullptr;
    if (!PyArg_Parse(_arg, "s", &nodeCls))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QString& subgName = self->subgIdx.data(ROLE_OBJNAME).toString();
    const QString& descName = QString::fromUtf8(nodeCls);
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    const QString& ident = NodesMgr::createNewNode(pModel, self->subgIdx, descName, QPointF(0, 0));
    //set value
    QModelIndex nodeIdx = pModel->nodeIndex(ident);
    QAbstractItemModel* pSubModel = const_cast<QAbstractItemModel*>(nodeIdx.model());
    if (!pSubModel)
    {
        PyErr_SetString(PyExc_Exception, "Subgraph is null");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    if (kw && PyDict_Check(kw))
    {
        PyObject* key, * value;
        Py_ssize_t pos = 0;
        INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        while (PyDict_Next(kw, &pos, &key, &value))
        {
            char* cKey = nullptr;
            if (!PyArg_Parse(key, "s", &cKey))
                continue;
            QString strKey= QString::fromUtf8(cKey);
            if (strKey == "view" || strKey == "mute" || strKey == "once")
            {
                int options = nodeIdx.data(ROLE_OPTIONS).toInt();
                if (strKey == "view")
                {
                    bool view = false;
                    if (PyArg_Parse(value, "b", &view) && view)
                        options |= OPT_VIEW;
                }
                else if (strKey == "mute")
                {
                    bool mute = false;
                    if (PyArg_Parse(value, "b", &mute) && mute)
                        options |= OPT_MUTE;
                }
                else if (strKey == "once")
                {
                    bool once = false;
                    if (PyArg_Parse(value, "b", &once) && once)
                        options |= OPT_ONCE;
                }
                pSubModel->setData(nodeIdx, options, ROLE_OPTIONS);
            }
            else if (strKey == "fold")
            {
                bool fold = false;
                if (PyArg_Parse(value, "b", &fold) && fold)
                    pSubModel->setData(nodeIdx, fold, ROLE_COLLASPED);
            }
            else if (strKey == "pos")
            {
                float x;
                float y;
                if (PyArg_ParseTuple(value, "ff", &x, &y))
                    pSubModel->setData(nodeIdx, QPointF(x, y), ROLE_OBJPOS);
            }
            else if (inputs.contains(strKey))
            {
                INPUT_SOCKET socket = inputs[strKey];
                QVariant val = parseValue(value, socket.info.type);
                if (!val.isValid())
                    continue;
                PARAM_UPDATE_INFO info;
                info.name = strKey;
                info.newValue = val;
                info.oldValue = socket.info.defaultValue;
                pModel->updateSocketDefl(socket.info.nodeid, info, self->subgIdx);
            }
            else if (params.contains(strKey))
            {
                PARAM_INFO param = params[strKey];
                QVariant val = parseValue(value, param.typeDesc);
                if (!val.isValid())
                    continue;
                PARAM_UPDATE_INFO info;
                info.name = strKey;
                info.newValue = val;
                info.oldValue = param.defaultValue;
                pModel->updateParamInfo(nodeIdx.data(ROLE_OBJID).toString(), info, self->subgIdx);
            }
        }
    }
    
    //const QModelIndex& nodeIdx = pModel->index(ident, self->subgIdx);
    std::string _subgName = subgName.toStdString();
    std::string _ident = ident.toStdString();
    PyObject* argList = Py_BuildValue("ss", _subgName.c_str(), _ident.c_str());

    PyObject* result = PyObject_CallObject((PyObject*)&ZNodeType, argList);
    Py_DECREF(argList);
    return result;
}

static PyObject*
Graph_deleteNode(ZSubGraphObject* self, PyObject* arg)
{
    PyObject* _arg = PyTuple_GET_ITEM(arg, 0);
    if (!PyUnicode_Check(_arg)) {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    char* ident = nullptr;
    if (!PyArg_Parse(_arg, "s", &ident))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    pModel->removeNode(ident, self->subgIdx);
    return Py_None;
}

static PyObject*
Graph_getNode(ZSubGraphObject* self, PyObject* arg)
{
    ZENO_HANDLE hGraph;
    //if (!PyArg_ParseTupleAndKeywords(arg, kwds, "i", kwList, &hGraph))
    //    return -1;
    PyObject* _arg = PyTuple_GET_ITEM(arg, 0);
    if (!PyUnicode_Check(_arg)) {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    char* _ident = nullptr;
    if (!PyArg_Parse(_arg, "s", &_ident))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

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
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QString& graphName = self->subgIdx.data(ROLE_OBJNAME).toString();
    const QString& outNode = QString::fromUtf8(_outNode);
    const QString& outSock = QString::fromUtf8(_outSock);
    const QString& inNode = QString::fromUtf8(_inNode);
    const QString& inSock = QString::fromUtf8(_inSock);

    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    QString inSockPath = UiHelper::constructObjPath(graphName, inNode, "[node]/inputs/", inSock);
    QModelIndex inSockeIdx =  pModel->indexFromPath(inSockPath);
    QString outSockPath = UiHelper::constructObjPath(graphName, outNode, "[node]/outputs/", outSock);
    QModelIndex outSockeIdx = pModel->indexFromPath(outSockPath);
    //dict panel.
    SOCKET_PROPERTY inProp = (SOCKET_PROPERTY)inSockeIdx.data(ROLE_PARAM_SOCKPROP).toInt();
    if (inProp & SOCKPROP_DICTLIST_PANEL)
    {
        QString inSockType = inSockeIdx.data(ROLE_PARAM_TYPE).toString();
        SOCKET_PROPERTY outProp = (SOCKET_PROPERTY)outSockeIdx.data(ROLE_PARAM_SOCKPROP).toInt();
        QString outSockType = outSockeIdx.data(ROLE_PARAM_TYPE).toString();
        QAbstractItemModel* pKeyObjModel =
            QVariantPtr<QAbstractItemModel>::asPtr(inSockeIdx.data(ROLE_VPARAM_LINK_MODEL));

        bool outSockIsContainer = false;
        if (inSockType == "list")
        {
            outSockIsContainer = outSockType == "list";
        }
        else if (inSockType == "dict")
        {
            const QModelIndex& fromNodeIdx = outSockeIdx.data(ROLE_NODE_IDX).toModelIndex();
            const QString& outNodeCls = fromNodeIdx.data(ROLE_OBJNAME).toString();
            const QString& outSockName = outSockeIdx.data(ROLE_PARAM_NAME).toString();
            outSockIsContainer = outSockType == "dict" || (outNodeCls == "FuncBegin" && outSockName == "args");
        }

        //if outSock is a container, connects it as usual.
        if (outSockIsContainer)
        {
            //legacy dict/list connection, and then we have to remove all inner dict key connection.
            ZASSERT_EXIT(pKeyObjModel, Py_None);
            for (int r = 0; r < pKeyObjModel->rowCount(); r++)
            {
                const QModelIndex& keyIdx = pKeyObjModel->index(r, 0);
                PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                for (QPersistentModelIndex _linkIdx : links)
                {
                    pModel->removeLink(_linkIdx, true);
                }
            }
        }
        else
        {
            //check multiple links
            QModelIndexList fromSockets;
            //check selected nodes.
            //model: ViewParamModel
            QString paramName = outSockeIdx.data(ROLE_PARAM_NAME).toString();
            QString paramType = outSockeIdx.data(ROLE_PARAM_TYPE).toString();
            QString toSockName = inSockeIdx.data(ROLE_OBJPATH).toString();

            // link to inner dict key automatically.
            int n = pKeyObjModel->rowCount();
            pModel->addExecuteCommand(
                new DictKeyAddRemCommand(true, pModel, inSockeIdx.data(ROLE_OBJPATH).toString(), n));
            inSockeIdx = pKeyObjModel->index(n, 0);
        }
    }

    //remove the edge in inNode:inSock, if exists.
    if (inProp != SOCKPROP_MULTILINK)
    {
        QPersistentModelIndex linkIdx;
        const PARAM_LINKS& links = inSockeIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
        if (!links.isEmpty())
            linkIdx = links[0];
        if (linkIdx.isValid())
            pModel->removeLink(linkIdx, true);
    }
    pModel->addLink(self->subgIdx, outSockeIdx, inSockeIdx);
    return Py_None;
}

static PyObject*
Graph_removeLink(ZSubGraphObject* self, PyObject* arg)
{
    char* _outNode, * _outSock, * _inNode, * _inSock;
    if (!PyArg_ParseTuple(arg, "ssss", &_outNode, &_outSock, &_inNode, &_inSock))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    const QString& graphName = self->subgIdx.data(ROLE_OBJNAME).toString();
    const QString& outNode = QString::fromUtf8(_outNode);
    const QString& outSock = QString::fromUtf8(_outSock);
    const QString& inNode = QString::fromUtf8(_inNode);
    const QString& inSock = QString::fromUtf8(_inSock);

    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    QModelIndex outIdx = pModel->nodeIndex(outNode);
    if (!outIdx.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Node '' is invalid").arg(outNode).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QModelIndex inIdx = pModel->nodeIndex(inNode);
    if (!inIdx.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Node '' is invalid").arg(inNode).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    QModelIndex subgIdx = pModel->index(graphName);
    if (!subgIdx.isValid())
    {
        PyErr_SetString(PyExc_Exception, QString("Subgraph '' is invalid").arg(graphName).toUtf8());
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    QModelIndex linkIdx = pModel->linkIndex(subgIdx,
                                            outIdx.data(ROLE_OBJID).toString(),
                                            outSock,
                                            inIdx.data(ROLE_OBJID).toString(),
                                            inSock);
    pModel->removeLink(linkIdx);
    return Py_None;
}

static PyMethodDef GraphMethods[] = {
    {"name", (PyCFunction)Graph_name, METH_NOARGS, "Return the name of graph"},
    {"createNode", (PyCFunction)Graph_createNode, METH_VARARGS|METH_KEYWORDS, "Add the node to this graph"},
    {"deleteNode", (PyCFunction)Graph_deleteNode, METH_VARARGS, "delete the node to this graph"},
    {"node", (PyCFunction)Graph_getNode, METH_VARARGS, "Get the node from the graph"},
    {"addLink", (PyCFunction)Graph_addLink, METH_VARARGS, "Add link"},
    {"removeLink", (PyCFunction)Graph_removeLink, METH_VARARGS, "remove link"},
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
#endif