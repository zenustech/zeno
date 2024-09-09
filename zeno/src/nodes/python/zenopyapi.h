#ifndef __ZENO_PYTHON_API_H__
#define __ZENO_PYTHON_API_H__

#ifdef ZENO_WITH_PYTHON

#include <Python.h>
#include <zeno/core/Graph.h>
#include "zeno_types/reflect/reflection.generated.hpp"

namespace zeno {

    typedef struct {
        PyObject_HEAD
        PyObject* first;
        PyObject* last;
        int number;
    } CustomObject;

    typedef struct {
        PyObject_HEAD
        std::weak_ptr<Graph> subgIdx;
    } ZSubGraphObject;

    typedef struct {
        PyObject_HEAD
        std::weak_ptr<Graph> subgIdx;
        std::weak_ptr<INode> nodeIdx;
    } ZNodeObject;

    zeno::reflect::Any parseValueFromPyObject(PyObject* v, const ParamType type);
}

extern PyTypeObject ZNodeType;
extern PyTypeObject SubgraphType;
#endif
#endif