#pragma once

#include <zeno/utils/defs.h>
#include <variant>
#include <string>
#include <memory>

namespace zeno {

using IValue = std::variant<std::string, int, float>;

struct IObject {
#ifndef ZENO_APIFREE
    ZENO_API IObject();
    ZENO_API virtual ~IObject();

    ZENO_API virtual std::shared_ptr<IObject> clone() const;
    ZENO_API virtual bool assign(IObject *other);
    ZENO_API virtual void dumpfile(std::string const &path);
#else
    virtual ~IObject() = default;
    virtual std::shared_ptr<IObject> clone() const { return nullptr; }
    virtual bool assign(IObject *other) { return false; }
    virtual void dumpfile(std::string const &path) {}
#endif

    template <class T>
    [[deprecated("use std::make_shared<T>")]]
    static std::shared_ptr<T> make() { return std::make_shared<T>(); }

    template <class T>
    [[deprecated("use dynamic_cast<T *>")]]
    T *as() { return dynamic_cast<T *>(this); }

    template <class T>
    [[deprecated("use dynamic_cast<const T *>")]]
    const T *as() const { return dynamic_cast<const T *>(this); }
};

template <class Derived, class Base = IObject>
struct IObjectClone : Base {
    virtual std::shared_ptr<IObject> clone() const {
        return std::make_shared<Derived>(static_cast<Derived const &>(*this));
    }

    virtual bool assign(IObject *other) {
        auto src = dynamic_cast<Derived *>(other);
        if (!src)
            return false;
        auto dst = static_cast<Derived *>(this);
        *dst = *src;
        return true;
    }
};

}
