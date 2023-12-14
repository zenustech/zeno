#ifndef __ZENO_PYTHON_API_H__
#define __ZENO_PYTHON_API_H__

#include <Python.h>
#include <QtWidgets>

typedef struct {
    PyObject_HEAD
    PyObject* first;
    PyObject* last;
    int number;
} CustomObject;

typedef struct {
    PyObject_HEAD
    QPersistentModelIndex subgIdx;
} ZSubGraphObject;

typedef struct {
    PyObject_HEAD
    QPersistentModelIndex subgIdx;
    QPersistentModelIndex nodeIdx;
} ZNodeObject;

extern PyTypeObject ZNodeType;
extern PyTypeObject SubgraphType;

#endif