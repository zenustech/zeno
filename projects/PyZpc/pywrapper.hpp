#pragma once
// ref: https://docs.python.org/3/c-api/intro.html#include-files
#define PY_SSIZE_T_CLEAN
#include <Python.h>
//
#include "zensim/ZpcMeta.hpp"

namespace zeno {

/// tags
enum struct py_object_e {
    unknown = 0,
    py_string,
    py_long,
    py_tuple,
    py_func
};
using py_obj_tag = zs::wrapv<py_object_e::unknown>;
using py_string_tag = zs::wrapv<py_object_e::py_string>;
using py_tuple_tag = zs::wrapv<py_object_e::py_tuple>;
using py_func_tag = zs::wrapv<py_object_e::py_func>;

constexpr auto py_obj_c = py_obj_tag{};
constexpr auto py_string_c = py_string_tag{};
constexpr auto py_tuple_c = py_tuple_tag{};
constexpr auto py_func_c = py_func_tag{};

struct pyobj {
    pyobj() = default;

    pyobj(PyObject *ptr) : obj{ptr} {
    }
    // pyobj(py_string_tag) {
    // }
    pyobj(py_tuple_tag, int n) {
        obj = PyTuple_New(n);
    }

    pyobj(pyobj &&o) {
        pyobj tmp{};
        obj = o.get();
        o = tmp;
    }
    pyobj &operator=(pyobj &&o) {
        pyobj tmp{};
        obj = o.get();
        o = tmp;
        return *this;
    }
    pyobj(const pyobj &o) {
        obj = const_cast<PyObject *>(o.get());
        Py_XINCREF(obj);
    }
    pyobj &operator=(const pyobj &o) {
        obj = const_cast<PyObject *>(o.get());
        Py_XINCREF(obj);
        return *this;
    }

    ~pyobj() {
        Py_XDECREF(obj);
    }

    auto operator->() noexcept {
        return get();
    }
    auto operator->() const noexcept {
        return get();
    }

    PyObject *get() noexcept {
        return obj;
    }
    const PyObject *get() const noexcept {
        return obj;
    }

    explicit operator bool() const noexcept {
        return obj != nullptr;
    }
    operator PyObject *() noexcept {
        return obj;
    }
    operator const PyObject *() const noexcept {
        return obj;
    }

  private:
    PyObject *obj{nullptr};
};

} // namespace zeno