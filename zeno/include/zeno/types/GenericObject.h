#pragma once

#include <zeno/core/IObject.h>
#include <utility>
#include <any>

namespace zeno {

//
// usage example:
//
// #include <zeno/types/UserData.h>
// #include <zeno/types/GenericObject.h>
//
// prim->userData().set("mydat", std::make_shared<GenericObject< MyData >>(mydat));
// prim->userData().set("functor", std::make_shared<GenericObject< std::function<void(int, std::string)> >>(functor));
//
// MyData &mydat = prim->userData().get<GenericObject< MyData >>("mydat")->get();
// auto &functor = prim->userData().get<GenericObject< std::function<void(int, std::string)> >>("functor")->get();
// functor(42, "yes");
//

template <class T = std::any>
struct GenericObject : IObjectClone<GenericObject<T>> {
    T value;

    GenericObject() = default;
    GenericObject(T const &value) : value(value) {}
    GenericObject(T &&value) : value(std::move(value)) {}

    T const &get() const {
        return value;
    }

    T &get() {
        return value;
    }

    T const &operator*() const {
        return value;
    }

    T &operator*() {
        return value;
    }

    T const *operator->() const {
        return std::addressof(value);
    }

    T *operator->() {
        return std::addressof(value);
    }

    void set(T const &x) {
        value = x;
    }

    void set(T &&x) {
        value = std::move(x);
    }
};

}
