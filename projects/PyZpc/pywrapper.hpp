#pragma once
// ref: https://docs.python.org/3/c-api/intro.html#include-files
#define PY_SSIZE_T_CLEAN
#include <Python.h>
//
#include "zensim/ZpcMeta.hpp"
#include "zensim/zpc_tpls/fmt/format.h"
#include <exception>

namespace zeno {

struct pyobj;
/// tags
enum struct py_object_e {
    py_obj = 0,
    py_string,
    py_long,
    py_tuple,
    py_dict,
    py_module,
    py_func
};
#define ZENO_PY_OBJ_UTILS(name)                                \
    using py_##name##_tag = zs::wrapv<py_object_e::py_##name>; \
    constexpr auto py_##name##_c = py_##name##_tag{};          \
    template <typename... Args>                                \
    auto pymake_##name(Args &&...args) {                       \
        return pyobj{py_##name##_c, FWD(args)...};             \
    }

ZENO_PY_OBJ_UTILS(obj)
ZENO_PY_OBJ_UTILS(string)
ZENO_PY_OBJ_UTILS(long)
ZENO_PY_OBJ_UTILS(tuple)
ZENO_PY_OBJ_UTILS(dict)
ZENO_PY_OBJ_UTILS(module)
ZENO_PY_OBJ_UTILS(func)

struct pyobj {
    pyobj() = default;

    pyobj(PyObject *ptr) {
        Py_XDECREF(obj);
        obj = ptr;
    }
    /// @brief string
    pyobj(py_string_tag, const std::string_view str) {
        obj = PyUnicode_InternFromString(str.data());
    }
    /// @brief long
    pyobj(py_long_tag, long n) {
        obj = PyLong_FromLong(n);
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("cannot construct long from long {}.\n", n));
    }
    pyobj(py_long_tag, long long n) {
        obj = PyLong_FromLongLong(n);
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("cannot construct long from long long {}.\n", n));
    }
    pyobj(py_long_tag, unsigned long n) {
        obj = PyLong_FromUnsignedLong(n);
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("cannot construct long from unsigned long {}.\n", n));
    }
    pyobj(py_long_tag, unsigned long long n) {
        obj = PyLong_FromUnsignedLongLong(n);
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("cannot construct long from unsigned long long {}.\n", n));
    }
    pyobj(py_long_tag, void *ptr) {
        obj = PyLong_FromVoidPtr(ptr);
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("cannot construct long from void ptr {}.\n", ptr));
    }
    /// @brief tuple
    pyobj(py_tuple_tag, int n) {
        obj = PyTuple_New(n);
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("failed to new a py tuple of size {}.\n", n));
    }
    /// @brief dict
    pyobj(py_dict_tag) {
        obj = PyDict_New();
        if (obj == nullptr)
            throw std::runtime_error("failed to new a py dict\n");
    }
    /// @brief module
    pyobj(py_module_tag, const pyobj &name) {
        obj = PyImport_Import(name);
        if (PyErr_Occurred())
            PyErr_Print();
    }
    /// @brief func
    pyobj(py_func_tag, const pyobj &module, const std::string_view funcName) {
        obj = PyObject_GetAttrString(module, funcName.data());
        if (obj == nullptr)
            throw std::runtime_error(fmt::format("failed to find func [{}] from module.\n", funcName));
        else if (!PyCallable_Check(obj)) {
            throw std::runtime_error(fmt::format("[{}] is not a callable of this module.\n", funcName));
        }
    }

    /// @brief mimic shared_ptr behavior
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
        obj = o.get();
        Py_XINCREF(obj);
    }
    pyobj &operator=(const pyobj &o) {
        obj = o.get();
        Py_XINCREF(obj);
        return *this;
    }

    ~pyobj() {
        Py_XDECREF(obj);
    }

    auto operator->() const noexcept {
        return get();
    }

    PyObject *get() const noexcept {
        return const_cast<PyObject *>(obj);
    }

    explicit operator bool() const noexcept {
        return obj != nullptr;
    }
    explicit operator long() const noexcept {
        return PyLong_AsLong(obj);
    }
    operator PyObject *() const noexcept {
        return const_cast<PyObject *>(obj);
    }

    mutable PyObject *obj{nullptr};
};

} // namespace zeno