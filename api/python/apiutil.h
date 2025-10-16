#pragma once

#ifndef __API_UTIL_H__
#define __API_UTIL_H__

#ifdef ZENO_PYAPI_STATICLIB
#include <Python.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zeno/core/data.h>

namespace py = pybind11;

namespace zpyapi {
    zeno::vecvar pylist2vec(py::list lst);
    py::list vec2pylist(zeno::vec3f vec);
}

#endif