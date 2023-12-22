#include "zenopyapi.h"
#include <QtWidgets>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/enum.h>
#include <zenomodel/include/nodesmgr.h>


static int
Node_init(ZNodeObject* self, PyObject* args, PyObject* kwds)
{
    char* _subgName, * _ident;
    if (!PyArg_ParseTuple(args, "ss", &_subgName, &_ident))
    {
        PyErr_SetString(PyExc_Exception, "args error");
        PyErr_WriteUnraisable(Py_None);
        return -1;
    }

    QString graphName = QString::fromUtf8(_subgName);
    QString ident = QString::fromUtf8(_ident);

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Current Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return -1;
    }

    self->subgIdx = pModel->index(graphName);
    self->nodeIdx = pModel->index(ident, self->subgIdx);
    return 0;
}

static PyObject*
Node_name(ZNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    const QString& name = self->nodeIdx.data(ROLE_CUSTOM_OBJNAME).toString();
    return PyUnicode_FromFormat(name.toUtf8());
}

static PyObject*
Node_class(ZNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    const QString& name = self->nodeIdx.data(ROLE_OBJNAME).toString();
    return PyUnicode_FromFormat(name.toUtf8());
}

static PyObject*
Node_ident(ZNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    const QString& name = self->nodeIdx.data(ROLE_OBJID).toString();
    return PyUnicode_FromFormat(name.toUtf8());
}

static PyObject*
Node_getattr(ZNodeObject* self, char* name)
{
    if (strcmp(name, "pos") == 0) {
        QPointF pos = self->nodeIdx.data(ROLE_OBJPOS).toPointF();
        PyObject* postuple = Py_BuildValue("dd", pos.x(), pos.y());
        return postuple;
    }
    if (strcmp(name, "ident") == 0) {
        std::string ident = self->nodeIdx.data(ROLE_OBJID).toString().toStdString();
        PyObject* value = Py_BuildValue("s", ident.c_str());
        return value;
    }
    if (strcmp(name, "objCls") == 0) {
        std::string cls = self->nodeIdx.data(ROLE_OBJNAME).toString().toStdString();
        PyObject* value = Py_BuildValue("s", cls.c_str());
        return value;
    }
    if (strcmp(name, "name") == 0) {
        std::string name = self->nodeIdx.data(ROLE_CUSTOM_OBJNAME).toString().toStdString();
        PyObject* value = Py_BuildValue("s", name.c_str());
        return value;
    }
    if (strcmp(name, "view") == 0 || strcmp(name, "mute") == 0 || strcmp(name, "once") == 0) {
        int opt = strcmp(name, "view") == 0 ? OPT_VIEW : strcmp(name, "mute") == 0 ? OPT_MUTE : OPT_ONCE;
        bool bOn = self->nodeIdx.data(ROLE_OPTIONS).toInt() & opt;
        PyObject* value = Py_BuildValue("b", bOn);
        return value;
    }
    return Py_None;
}

static QVariant parseValue(PyObject *v, const QString& type)
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
static int
Node_setattr(ZNodeObject* self, char* name, PyObject* v)
{
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(self->nodeIdx.model());
    if (!pModel)
    {
        PyErr_SetString(PyExc_Exception, "Model is NULL");
        PyErr_WriteUnraisable(Py_None);
        return -1;
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

        pModel->setData(self->nodeIdx, QPointF(x, y), ROLE_OBJPOS);
    }
    else if (strcmp(name, "view") == 0 || strcmp(name, "mute") == 0 || strcmp(name, "once") == 0)
    {
        bool bOn;
        if (!PyArg_Parse(v, "b", &bOn))
        {
            PyErr_SetString(PyExc_Exception, "args error");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        int options_old = self->nodeIdx.data(ROLE_OPTIONS).toInt();
        int options = options_old;
        int opt = strcmp(name, "view") == 0 ? OPT_VIEW : strcmp(name, "mute") == 0 ? OPT_MUTE : OPT_ONCE;
        if (bOn) {
            options |= opt;
        }
        else if (options & opt) {
            options ^= opt;
        }
        pModel->setData(self->nodeIdx, options, ROLE_OPTIONS);
    }
    else if (strcmp(name, "fold") == 0)
    {
        bool bOldCollasped = self->nodeIdx.data(ROLE_COLLASPED).toBool();
        bool bCollasped;
        if (!PyArg_Parse(v, "b", &bCollasped))
        {
            PyErr_SetString(PyExc_Exception, "args error");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        pModel->setData(self->nodeIdx, bCollasped, ROLE_COLLASPED);
    }
    else
    {
        INPUT_SOCKETS inputs = self->nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        PARAMS_INFO params = self->nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        PARAM_UPDATE_INFO info;
        info.name = name;
        IGraphsModel* pCurrModel = GraphsManagment::instance().currentModel();
        if (!pCurrModel)
        {
            PyErr_SetString(PyExc_Exception, "Model is NULL");
            PyErr_WriteUnraisable(Py_None);
            return -1;
        }
        if (inputs.contains(name))
        {
            INPUT_SOCKET socket = inputs[name];
            QVariant val = parseValue(v, socket.info.type);
            if (!val.isValid())
                return -1;
            info.newValue = val;
            info.oldValue = socket.info.defaultValue;
            pCurrModel->updateSocketDefl(socket.info.nodeid, info, self->subgIdx);
        }
        
        else if (params.contains(name))
        {
            PARAM_INFO param = params[name];
            QVariant val = parseValue(v, param.typeDesc);
            if (!val.isValid())
                return -1;
            info.newValue = val;
            info.oldValue = param.defaultValue;
            pCurrModel->updateParamInfo(self->nodeIdx.data(ROLE_OBJID).toString(), info, self->subgIdx);
        }

    }
    return 0;
}


//node methods.
static PyMethodDef NodeMethods[] = {
    {"name",  (PyCFunction)Node_name, METH_NOARGS, "Return the name of node"},
    {"objCls", (PyCFunction)Node_class, METH_NOARGS, "Return the class of node"},
    {"ident", (PyCFunction)Node_ident, METH_NOARGS, "Return the ident of node"},
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