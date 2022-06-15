#pragma once

#include <zeno/utils/api.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/types/IObjectXMacro.h>
#include <string>
#include <memory>
#include <any>

namespace zeno {

struct UserData;
struct IObjectVisitor;

struct IObject {
    using polymorphic_base_type = IObject;

    mutable std::any m_userData;
    //std::string nodeid;

#ifndef ZENO_APIFREE
    ZENO_API IObject();
    ZENO_API IObject(IObject const &);
    ZENO_API IObject(IObject &&);
    ZENO_API IObject &operator=(IObject const &);
    ZENO_API IObject &operator=(IObject &&);
    ZENO_API virtual ~IObject();

    ZENO_API virtual std::shared_ptr<IObject> clone() const;
    ZENO_API virtual std::shared_ptr<IObject> move_clone();
    ZENO_API virtual bool assign(IObject const *other);
    ZENO_API virtual bool move_assign(IObject const *other);
    ZENO_API virtual void accept(IObjectVisitor *visitor);

    ZENO_API UserData &userData() const;
#else
    virtual ~IObject() = default;
    virtual std::shared_ptr<IObject> clone() const { return nullptr; }
    virtual std::shared_ptr<IObject> move_clone() { return nullptr; }
    virtual bool assign(IObject *other) { return false; }
    virtual bool move_assign(IObject *other) { return false; }
    ZENO_API virtual void accept(IObjectVisitor *visitor) {}

    UserData &userData() { return *reinterpret_cast<UserData *>(0); }
#endif

    template <class T>
    [[deprecated("use std::make_shared<T>")]]
    static std::shared_ptr<T> make() { return std::make_shared<T>(); }

    template <class T>
    [[deprecated("use dynamic_cast<T *>")]]
    T *as() { return zeno::safe_dynamic_cast<T>(this); }

    template <class T>
    [[deprecated("use dynamic_cast<const T *>")]]
    const T *as() const { return zeno::safe_dynamic_cast<T>(this); }
};


#define _ZENO_PER_XMACRO(TypeName, ...) \
struct TypeName;
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO


struct IObjectVisitor {
#ifdef ZENO_APIFREE
#define _ZENO_PER_XMACRO(TypeName, ...) \
    virtual void visit(TypeName *object) { \
        visit(reinterpret_cast<IObject *>(object)); \
    }
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO

    virtual void visit(IObject *object) {
    }

    virtual ~IObjectVisitor() = default;
#else
#define _ZENO_PER_XMACRO(TypeName, ...) \
    ZENO_API virtual void visit(TypeName *object);
ZENO_XMACRO_IObject(_ZENO_PER_XMACRO)
#undef _ZENO_PER_XMACRO

    ZENO_API virtual void visit(IObject *object);

    ZENO_API virtual ~IObjectVisitor();
#endif
};


template <class Derived, class CustomBase = IObject>
struct IObjectClone : CustomBase {
    //using has_iobject_clone = std::true_type;

    virtual void accept(IObjectVisitor *visitor) override {
        visitor->visit(static_cast<Derived *>(this));
    }

    virtual std::shared_ptr<IObject> clone() const override {
        return std::make_shared<Derived>(static_cast<Derived const &>(*this));
    }

    virtual std::shared_ptr<IObject> move_clone() override {
        return std::make_shared<Derived>(static_cast<Derived &&>(*this));
    }

    virtual bool assign(IObject const *other) override {
        auto src = dynamic_cast<Derived const *>(other);
        if (!src)
            return false;
        auto dst = static_cast<Derived *>(this);
        *dst = *src;
        return true;
    }

    virtual bool move_assign(IObject const *other) override {
        auto src = dynamic_cast<Derived const *>(other);
        if (!src)
            return false;
        auto dst = static_cast<Derived *>(this);
        *dst = std::move(*src);
        return true;
    }
};

using zany = std::shared_ptr<IObject>;

}
