#pragma once


#include <zeno/dop/Node.h>
#include <zeno/dop/Descriptor.h>
#include "utils/safe_at.h"
#include "utils/safe_dynamic_cast.h"
#include "utils/UserData.h"



namespace zeno {


struct IObject {
    using polymorphic_base_type = IObject;

    UserData userData;

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


struct INode : ZENO_NAMESPACE::dop::Node {
protected:
    virtual void apply() = 0;

    bool has_input2(std::string const &id) const;
    Any get_input2(std::string const &id) const;
    void set_output2(std::string const &id, Any &&obj);
    void _set_output2(std::string const &name, Any &&val);

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

    /* todo: deprecated */
    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id, typeid(T).name());
        return safe_dynamic_cast<T>(std::move(obj), "input socket `" + id + "` ");
    }

    /* todo: deprecated */
    auto get_param(std::string const &id) const {
        std::variant<int, float, std::string> res;
        auto inpid = id + ":";
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


struct ParamDescriptor {
  std::string type, name, defl;

  ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
};

struct SocketDescriptor {
  std::string type, name, defl;

  SocketDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl = {})
      : type(type), name(name), defl(defl) {}

  //[[deprecated("use {\"sockType\", \"sockName\"} instead of \"sockName\"")]]
  SocketDescriptor(const char *name)
      : SocketDescriptor({}, name) {}
};

struct Descriptor {
  std::vector<SocketDescriptor> inputs;
  std::vector<SocketDescriptor> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;
};

void defNodeClass(std::function<std::unique_ptr<ZENO_NAMESPACE::dop::Node>()> func,
        std::string const &name, Descriptor const &desc);

template <class T>
int defNodeClass(std::string const &name, Descriptor const &desc) {
    defNodeClass(std::make_unique<T>, name, desc);
    return 1;
}

template <class F>
auto defNodeClassHelper(F const &func, std::string const &name) {
    return [=] (Descriptor const &desc) -> int {
        defNodeClass(func, name, desc);
        return 1;
    };
}

#define ZENO_DEFNODE(Class) \
    static int def##Class = ::zeno::defNodeClassHelper(std::make_unique<Class>, #Class)

#define ZENO_DEFOVERLOADNODE(Class, PostFix, ...) \
    static int def##Class##PostFix = ::zeno::defOverloadNodeClassHelper(std::make_unique<Class##PostFix>, #Class, {__VA_ARGS__})

#define ZENDEFNODE(Class, ...) \
    ZENO_DEFNODE(Class)(__VA_ARGS__)


template <class T>
using SharedPtr = std::shared_ptr<T>;

template <class T, class ...Ts>
auto makeShared(Ts &&...ts) {
    return std::make_shared<T>(std::forward<Ts>(ts)...);
}


}
