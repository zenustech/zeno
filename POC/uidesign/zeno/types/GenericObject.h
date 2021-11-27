#pragma once

#include <zeno/core/IObject.h>

namespace zeno {

template <class T>
struct GenericObject : IObjectClone<GenericObject<T>> {
    T obj;

    GenericObject() = default;
    GenericObject(T const &t) : obj(t) {}

    T &get() { return obj; }
    T const &get() const { return obj; }
    void set(T const &t) { obj = t; }

    auto &operator=(T const &obj) { set(obj); return *this; }

    T *operator*() { return get(); }
    T *operator*() const { return get(); }
    T const *operator->() const { return &get(); }
    T *operator->() { return &get(); }
    operator T &() { return get(); }
    operator T const &() const { return get(); }
};

}
