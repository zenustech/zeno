#ifdef ZENO_WITH_PYTHON
#include "zenopyapi.h"

namespace zeno {

static int
Node_init(ZNodeObject* self, PyObject* args, PyObject* kwds)
{
    char* uuid_path;
    if (!PyArg_ParseTuple(args, "s", &uuid_path))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return -1;
    }

    auto mainGraph = zeno::getSession().mainGraph;
    if (!mainGraph)
    {
        PyErr_SetString(PyExc_Exception, "Current main graph is NULL");
        PyErr_WriteUnraisable(Py_None);
        return -1;
    }

    std::shared_ptr<INode> spNode = mainGraph->getNodeByUuidPath(std::string(uuid_path));

    self->subgIdx = spNode->getGraph();
    self->nodeIdx = spNode;
    return 0;
}

static PyObject*
Node_name(ZNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    std::shared_ptr<INode> spNode = self->nodeIdx.lock();
    if (!spNode) {
        PyErr_SetString(PyExc_Exception, "Current node is NULL");
        return 0;
    }
    const std::string& name = spNode->get_name();
    return PyUnicode_FromFormat(name.c_str());
}

static PyObject*
Node_class(ZNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    std::shared_ptr<INode> spNode = self->nodeIdx.lock();
    if (!spNode) {
        PyErr_SetString(PyExc_Exception, "Current node is NULL");
        return 0;
    }
    const std::string& cls = spNode->get_nodecls();
    return PyUnicode_FromFormat(cls.c_str());
}

static PyObject* buildValue(const zeno::reflect::Any& value, const ParamType type)
{
    if (type == zeno::types::gParamType_String)
    {
        return Py_BuildValue("s", zeno::reflect::any_cast<std::string>(value).c_str());
    }
    else if (type == zeno::types::gParamType_Int)
    {
        return Py_BuildValue("i", zeno::reflect::any_cast<int>(value));
    }
    else if (type == zeno::types::gParamType_Float)
    {
        return Py_BuildValue("f", zeno::reflect::any_cast<float>(value));
    }
#if 0
    else if (type.startsWith("vec"))
    {
        const auto& vec = value.value<UI_VECTYPE>();
        QString format;
        
        if (type.contains("i"))
        {
            for (int i = 0; i < vec.size(); i++)
            {
                format += "i";
            }
        }
        else if (type.contains("f"))
        {
            for (int i = 0; i < vec.size(); i++)
            {
                format += "f";
            }
        }
        if (vec.size() == 2)
        {
            return Py_BuildValue(format.toUtf8(), vec[0], vec[1]);
        }
        else if (vec.size() == 3)
        {
            return Py_BuildValue(format.toUtf8(), vec[0], vec[1], vec[2]);
        }
        else if (vec.size() == 4)
        {
            return Py_BuildValue(format.toUtf8(), vec[0], vec[1], vec[2], vec[3]);
        }
    }
#endif
    PyErr_SetString(PyExc_Exception, "build value failed");
    PyErr_WriteUnraisable(Py_None);
    return Py_None;
}

static PyObject*
Node_getattr(ZNodeObject* self, char* name)
{
    std::shared_ptr<INode> spNode = self->nodeIdx.lock();
    if (!spNode) {
        PyErr_SetString(PyExc_Exception, "Current node is NULL");
        return 0;
    }

    if (strcmp(name, "pos") == 0) {
        auto pos = spNode->get_pos();
        PyObject* postuple = Py_BuildValue("dd", pos.first, pos.second);
        return postuple;
    }
    else if (strcmp(name, "objCls") == 0) {
        std::string cls = spNode->get_nodecls();
        PyObject* value = Py_BuildValue("s", cls.c_str());
        return value;
    }
    else if (strcmp(name, "name") == 0) {
        std::string name = spNode->get_name();
        PyObject* value = Py_BuildValue("s", name.c_str());
        return value;
    }
    else if (strcmp(name, "view") == 0) {
        bool bOn = spNode->is_view();
        PyObject* value = Py_BuildValue("b", bOn);
        return value;
    }
    else if (strcmp(name, "mute") == 0) {
        bool bOn = false;   //TODO: mute
        PyObject* value = Py_BuildValue("b", bOn);
        return value;
    }
    else {
        bool bExisted = false;
        zeno::ParamPrimitive prim = spNode->get_input_prim_param(name, &bExisted);
        if (!bExisted) {
            PyErr_SetString(PyExc_Exception, "build value failed");
            PyErr_WriteUnraisable(Py_None);
            return Py_None;
        }
        return buildValue(prim.defl, prim.type);
    }
    return Py_None;
}

static int
Node_setattr(ZNodeObject* self, char* name, PyObject* v)
{
    std::shared_ptr<INode> spNode = self->nodeIdx.lock();
    if (!spNode) {
        PyErr_SetString(PyExc_Exception, "Current node is NULL");
        return 0;
    }

    if (strcmp(name, "pos") == 0)
    {
        float x, y;
        if (!PyArg_ParseTuple(v, "ff", &x, &y))
        {
            PyErr_SetString(PyExc_Exception, "args error");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        spNode->set_pos({ x,y });
    }
    else if (strcmp(name, "view") == 0) {
        bool bOn;
        if (!PyArg_Parse(v, "b", &bOn))
        {
            PyErr_SetString(PyExc_Exception, "args error");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        spNode->set_view(bOn);
    }
    else if (strcmp(name, "view") == 0 || strcmp(name, "mute") == 0)
    {
        bool bOn;
        if (!PyArg_Parse(v, "b", &bOn))
        {
            PyErr_SetString(PyExc_Exception, "args error");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        //spNode->set_mute(bOn);
    }
    else
    {
        bool bExisted = false;
        zeno::ParamPrimitive prim = spNode->get_input_prim_param(name, &bExisted);
        if (!bExisted) {
            PyErr_SetString(PyExc_Exception, "build value failed");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        auto newVal = parseValueFromPyObject(v, prim.type);
        spNode->update_param(name, newVal);
    }
    return 0;
}


//node methods.
static PyMethodDef NodeMethods[] = {
    {"name",  (PyCFunction)Node_name, METH_NOARGS, "Return the name of node"},
    {"objCls", (PyCFunction)Node_class, METH_NOARGS, "Return the class of node"},
    {NULL, NULL, 0, NULL}
};

PyTypeObject ZNodeType = {
    // clang-format off
        PyVarObject_HEAD_INIT(nullptr, 0)
        // clang-format on
        "zeno.Node",                        /* tp_name */
        sizeof(ZNodeObject),                /* tp_basicsize */
        0,                                  /* tp_itemsize */
        nullptr,                            /* tp_dealloc */
    #if PY_VERSION_HEX < 0x03080000
        nullptr,                            /* tp_print */
    #else
        0, /* tp_vectorcall_offset */
    #endif
        (getattrfunc)Node_getattr,          /* tp_getattr */
        (setattrfunc)Node_setattr,          /* tp_setattr */
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
        PyDoc_STR("Zeno Node objects"),     /* tp_doc */
        nullptr,                            /* tp_traverse */
        nullptr,                            /* tp_clear */
        nullptr,                            /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        nullptr,                            /* tp_iter */
        nullptr,                            /* tp_iternext */
        NodeMethods,                            /* tp_methods */
        nullptr,                            /* tp_members */
        nullptr,                            /* tp_getset */
        nullptr,                            /* tp_base */
        nullptr,                            /* tp_dict */
        nullptr,                            /* tp_descr_get */
        nullptr,                            /* tp_descr_set */
        0,                          /* tp_dictoffset */
       (initproc)Node_init,                /* tp_init */
        nullptr,                            /* tp_alloc */
        PyType_GenericNew,                  /* tp_new */
};

}

#endif