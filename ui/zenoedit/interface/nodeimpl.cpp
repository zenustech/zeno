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
        return -1;

    QString graphName = QString::fromUtf8(_subgName);
    QString ident = QString::fromUtf8(_ident);

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return 0;

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
    if (strcmp(name, "class") == 0) {
        std::string cls = self->nodeIdx.data(ROLE_OBJNAME).toString().toStdString();
        PyObject* value = Py_BuildValue("s", cls.c_str());
        return value;
    }
    if (strcmp(name, "name") == 0) {
        std::string name = self->nodeIdx.data(ROLE_CUSTOM_OBJNAME).toString().toStdString();
        PyObject* value = Py_BuildValue("s", name.c_str());
        return value;
    }
    return Py_None;
}

static int
Node_setattr(ZNodeObject* self, char* name, PyObject* v)
{
    if (strcmp(name, "pos") == 0)
    {
        float x, y;
        if (!PyArg_ParseTuple(v, "ff", &x, &y))
            return -1;

        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(self->nodeIdx.model());
        pModel->setData(self->nodeIdx, QPointF(x, y), ROLE_OBJPOS);
    }
    return 0;
}


//node methods.
static PyMethodDef NodeMethods[] = {
    {"name",  (PyCFunction)Node_name, METH_NOARGS, "Return the name of node"},
    {"class", (PyCFunction)Node_class, METH_NOARGS, "Return the class of node"},
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