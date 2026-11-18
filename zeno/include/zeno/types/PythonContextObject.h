#pragma once

//============== Python 上下文对象（链式 python 脚本传参/共享命名空间）===============
//
// SequentialPythonNode 保存在链私有 globals dict（PyObject*）上运行脚本的结果，
// 并输出到 context 端口。下游节点（如 ReadRobotSimulation）通过 get_input<PythonContextObject>
// 取得本对象，再调用 getGlobals() 读取 python 变量。因此该类型必须放进公共头文件，
// 供 PythonNode 与 Robot 系列节点共享同一 C++ 类型。

#include <zeno/zeno.h>

#ifdef ZENO_WITH_PYTHON3
#include <Python.h>

namespace zeno {

// 持有 python globals dict（PyObject*），跨节点传递脚本运行环境。
// 复制/赋值时对 globals 做引用计数管理，析构时释放。
struct PythonContextObject : IObjectClone<PythonContextObject> {
private:
    PyObject* globals = nullptr;

public:
    PythonContextObject() = default;

    PythonContextObject(PyObject* globals_) : globals(globals_) {
        Py_XINCREF(globals);
    }

    PythonContextObject(const PythonContextObject& other) : globals(other.globals) {
        Py_XINCREF(globals);
    }

    PythonContextObject(PythonContextObject&& other) noexcept : globals(other.globals) {
        other.globals = nullptr;
    }

    PythonContextObject& operator=(const PythonContextObject& other) {
        if (this != &other) {
            Py_XDECREF(globals);
            globals = other.globals;
            Py_XINCREF(globals);
        }
        return *this;
    }

    PythonContextObject& operator=(PythonContextObject&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(globals);
            globals = other.globals;
            other.globals = nullptr;
        }
        return *this;
    }

    ~PythonContextObject() {
        Py_XDECREF(globals);
    }

    PyObject* getGlobals() const {
        return globals;
    }
};

}

#endif