#ifdef ZENO_WITH_PYTHON
#include "zenopyapi.h"


namespace zeno{

zeno::reflect::Any parseValueFromPyObject(PyObject* v, const ParamType type)
{
    if (type == zeno::types::gParamType_String)
    {
        char* _val = nullptr;
        if (PyArg_Parse(v, "s", &_val))
        {
            return std::string(_val);
        }
    }
    else if (type == zeno::types::gParamType_Int)
    {
        int _val;
        if (PyArg_Parse(v, "i", &_val))
        {
            return _val;
        }
    }
    else if (type == zeno::types::gParamType_Float)
    {
        float _val;
        if (PyArg_Parse(v, "f", &_val))
        {
            return _val;
        }
    }
#if 0
    //TODO: vec type
    else if (type == zeno::types::)
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
#endif
    else
    {
        PyErr_SetString(PyExc_Exception, "not support type");
        PyErr_WriteUnraisable(Py_None);
        return zeno::reflect::Any();
    }
}

//init function
static int
Graph_init(ZSubGraphObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwList[] = { "hGraph", NULL };
    char* path;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwList, &path))
        return -1;

    std::string graphPath(path);
    auto mainGraph = zeno::getSession().mainGraph;
    if (!mainGraph)
    {
        PyErr_SetString(PyExc_Exception, "Current main graph is NULL");
        PyErr_WriteUnraisable(Py_None);
        return 0;
    }

    self->subgIdx = mainGraph->getGraphByPath(graphPath);
    return 0;
}

static PyObject*
Graph_name(ZSubGraphObject* self, PyObject* Py_UNUSED(ignored))
{
    if (!zeno::getSession().mainGraph)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }
    auto spGraph = self->subgIdx.lock();
    return PyUnicode_FromFormat(spGraph->getName().c_str());
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

    auto spGraph = self->subgIdx.lock();
    if (!spGraph)
    {
        PyErr_SetString(PyExc_Exception, "graph is null");
        PyErr_WriteUnraisable(Py_None);
        return Py_None;
    }

    std::shared_ptr<INode> spNode = spGraph->createNode(nodeCls);

    if (kw && PyDict_Check(kw))
    {
        PyObject* key, * value;
        Py_ssize_t pos = 0;

        PrimitiveParams input_prims = spNode->get_input_primitive_params();

        while (PyDict_Next(kw, &pos, &key, &value))
        {
            char* cKey = nullptr;
            if (!PyArg_Parse(key, "s", &cKey))
                continue;

            std::string strKey(cKey);
            if (strKey == "view" || strKey == "mute" || strKey == "once")
            {
                if (strKey == "view")
                {
                    bool view = false;
                    if (PyArg_Parse(value, "b", &view) && view)
                    {
                        spNode->set_view(true);
                    }
                }
                else if (strKey == "mute")
                {
                    bool mute = false;
                    if (PyArg_Parse(value, "b", &mute) && mute)
                    {
                        //TODO: mute
                        //spNode->set_mute(true);
                    }
                }
            }
            else if (strKey == "pos")
            {
                float x;
                float y;
                if (PyArg_ParseTuple(value, "ff", &x, &y)) {
                    spNode->set_pos({ x, y });
                }
            }
            else
            {
                bool bExist = true;
                ParamPrimitive primInfo = spNode->get_input_prim_param(strKey, &bExist);
                if (!bExist) {
                    //TODO:
                    return Py_None;
                }

                auto val = parseValueFromPyObject(value, primInfo.type);
                spNode->update_param(strKey, val);
            }
        }
    }

    std::string full_uuid_path = spNode->get_uuid_path();
    PyObject* argList = Py_BuildValue("s", full_uuid_path.c_str());
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

    auto mainGraph = zeno::getSession().mainGraph;
    if (!mainGraph)
    {
        PyErr_SetString(PyExc_Exception, "Current main graph is NULL");
        PyErr_WriteUnraisable(Py_None);
        return 0;
    }

    auto spGraph = self->subgIdx.lock();
    if (!spGraph) {
        PyErr_SetString(PyExc_Exception, "Current graph is NULL");
        return Py_None;
    }
    spGraph->removeNode(std::string(ident));
    return Py_None;
}

static PyObject*
Graph_getNode(ZSubGraphObject* self, PyObject* arg)
{
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

    auto spGraph = self->subgIdx.lock();
    if (!spGraph) {
        PyErr_SetString(PyExc_Exception, "Current graph is NULL");
        return Py_None;
    }

    auto spNode = spGraph->getNode(std::string(_ident));
    if (!spNode) {
        PyErr_SetString(PyExc_Exception, "Current Node is NULL");
        return Py_None;
    }

    std::string _uuidpath = spNode->get_uuid_path();
    PyObject* argList = Py_BuildValue("ss", _uuidpath.c_str());
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

    auto spGraph = self->subgIdx.lock();
    if (!spGraph) {
        PyErr_SetString(PyExc_Exception, "Current graph is NULL");
        return Py_None;
    }

    auto outNode = spGraph->getNode(std::string(_outNode));
    if (!outNode) {
        PyErr_SetString(PyExc_Exception, "OutNode is NULL");
        return Py_None;
    }

    auto inNode = spGraph->getNode(std::string(_inNode));
    if (!inNode) {
        PyErr_SetString(PyExc_Exception, "inNode is NULL");
        return Py_None;
    }

    EdgeInfo edge;
    edge.inNode = std::string(_inNode);
    edge.outNode = std::string(_outNode);
    edge.inParam = std::string(_inSock);
    edge.outParam = std::string(_outSock);
    spGraph->addLink(edge);
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

    auto spGraph = self->subgIdx.lock();
    if (!spGraph) {
        PyErr_SetString(PyExc_Exception, "Current graph is NULL");
        return Py_None;
    }

    auto outNode = spGraph->getNode(std::string(_outNode));
    if (!outNode) {
        PyErr_SetString(PyExc_Exception, "OutNode is NULL");
        return Py_None;
    }

    auto inNode = spGraph->getNode(std::string(_inNode));
    if (!inNode) {
        PyErr_SetString(PyExc_Exception, "inNode is NULL");
        return Py_None;
    }

    EdgeInfo edge;
    edge.inNode = std::string(_inNode);
    edge.outNode = std::string(_outNode);
    edge.inParam = std::string(_inSock);
    edge.outParam = std::string(_outSock);
    spGraph->removeLink(edge);
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

}

#endif
