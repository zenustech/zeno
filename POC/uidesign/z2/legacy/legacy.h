#pragma once


#include <z2/dop/Node.h>
#include <z2/dop/Descriptor.h>
#include "safe_at.h"
#include "safe_dynamic_cast.h"



namespace z2::legacy {


struct IObject {
    using polymorphic_base_type = IObject;

    IObject();
    virtual ~IObject();

    virtual std::shared_ptr<IObject> clone() const;
    virtual std::shared_ptr<IObject> move_clone();
    virtual bool assign(IObject *other);
    virtual bool move_assign(IObject *other);

    template <class T>
    [[deprecated("use std::make_shared<T>")]]
    static std::shared_ptr<T> make() { return std::make_shared<T>(); }

    template <class T>
    [[deprecated("use dynamic_cast<T *>")]]
    T *as() { return safe_dynamic_cast<T>(this); }

    template <class T>
    [[deprecated("use dynamic_cast<const T *>")]]
    const T *as() const { return safe_dynamic_cast<T>(this); }
};

template <class Derived, class Base = IObject>
struct IObjectClone : Base {
    using has_iobject_clone = std::true_type;

    virtual std::shared_ptr<IObject> clone() const override {
        return std::make_shared<Derived>(static_cast<Derived const &>(*this));
    }

    virtual std::shared_ptr<IObject> move_clone() override {
        return std::make_shared<Derived>(static_cast<Derived &&>(*this));
    }

    virtual bool assign(IObject *other) override {
        auto src = dynamic_cast<Derived *>(other);
        if (!src)
            return false;
        auto dst = static_cast<Derived *>(this);
        *dst = *src;
        return true;
    }

    virtual bool move_assign(IObject *other) override {
        auto src = dynamic_cast<Derived *>(other);
        if (!src)
            return false;
        auto dst = static_cast<Derived *>(this);
        *dst = std::move(*src);
        return true;
    }
};


struct INode : dop::Node {
protected:
    virtual void apply() = 0;

    bool has_input2(std::string const &id) const;
    std::any get_input2(std::string const &id) const;
    void set_output2(std::string const &id, std::any &&obj);

    /* todo: deprecated */
    bool has_input(std::string const &id) const;

    /* todo: deprecated */
    std::shared_ptr<IObject> get_input(std::string const &id, std::string const &msg = "IObject") const;

    /* todo: deprecated */
    void set_output(std::string const &id, std::shared_ptr<IObject> &&obj) {
        set_output2(id, std::move(obj));
    }

    template <class T>
    T get_input2(std::string const &id) const {
        return safe_any_cast<T>(get_input2(id), "input `" + id + "` ");
    }

    template <class T>
    bool has_input2(std::string const &id) const {
        if (!has_input2(id))
            return false;
        return silent_any_cast<T>(get_input2(id)).has_value();
    }

    /* todo: deprecated */
    template <class T>
    bool has_input(std::string const &id) const {
        if (!has_input(id))
            return false;
        if (!has_input2<std::shared_ptr<IObject>>(id))
            return false;
        auto obj = get_input(id);
        auto p = std::dynamic_pointer_cast<T>(std::move(obj));
        return (bool)p;
    }

    bool _implicit_cast_from_to(std::string const &id,
        std::shared_ptr<IObject> const &from, std::shared_ptr<IObject> const &to);

    /* todo: deprecated */
    template <class T>
    std::enable_if_t<!std::is_abstract_v<T> && std::is_trivially_constructible_v<T>,
    std::shared_ptr<T>> get_input(std::string const &id) const {
        auto obj = get_input(id, typeid(T).name());
        if (auto p = std::dynamic_pointer_cast<T>(obj); p) {
            return p;
        }
        auto ret = std::make_shared<T>();
        if (!const_cast<INode *>(this)->_implicit_cast_from_to(id, obj, ret)) {
            throw ztd::make_error("input socket `" + id + "` expect IObject of `"
                + typeid(T).name() + "`, got `" + typeid(*obj).name() + "` (get_input)");
        }
        return ret;
    }

    /* todo: deprecated */
    template <class T>
    std::enable_if_t<std::is_abstract_v<T> || !std::is_trivially_constructible_v<T>,
    std::shared_ptr<T>> get_input(std::string const &id) const {
        auto obj = get_input(id, typeid(T).name());
        return safe_dynamic_cast<T>(std::move(obj), "input socket `" + id + "` ");
    }

    /* todo: deprecated */
    auto get_param(std::string const &id) const {
        std::variant<int, float, std::string> res;
        auto inpid = id + ":";
        using scalar_type_variant = std::variant
            < bool
            , uint8_t
            , uint16_t
            , uint32_t
            , uint64_t
            , int8_t
            , int16_t
            , int32_t
            , int64_t
            , float
            , double
            >;
        if (has_input2<scalar_type_variant>(inpid)) {
            std::visit([&] (auto const &x) {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_integral_v<T>) {
                    res = (int)x;
                } else {
                    res = (float)x;
                }
            }, get_input2<scalar_type_variant>(inpid));
        } else {
            res = get_input2<std::string>(inpid);
        }
        return res;
    }

    /* todo: deprecated */
    template <class T>
    T get_param(std::string const &id) const {
        //return std::get<T>(get_param(id));
        return get_input2<T>(id + ":");
    }
};


}
