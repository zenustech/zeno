#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/Any.h>
#include <zeno/utils/Exception.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>

namespace zeno {

struct Graph;
struct INodeClass;
struct Scene;
struct Session;
struct GlobalState;

struct INode {
public:
    Graph *graph = nullptr;
    INodeClass *nodeClass = nullptr;

    std::string myname;
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, zany> inputs;
    std::map<std::string, zany> outputs;
    std::set<std::string> options;
    zany muted_output;

    ZENO_API INode();
    ZENO_API virtual ~INode();

    INode(INode const &) = delete;
    INode &operator=(INode const &) = delete;
    INode(INode &&) = delete;
    INode &operator=(INode &&) = delete;

    ZENO_API void doComplete();
    ZENO_API void doApply();

protected:
    ZENO_API bool checkApplyCondition();
    ZENO_API bool requireInput(std::string const &ds);

    ZENO_API virtual void preApply();
    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;

    ZENO_API Graph *getThisGraph() const;
    ZENO_API Scene *getThisScene() const;
    ZENO_API Session *getThisSession() const;
    ZENO_API GlobalState *getGlobalState() const;

    ZENO_API bool has_option(std::string const &id) const;
    ZENO_API bool has_input2(std::string const &id) const;
    ZENO_API zany get_input2(std::string const &id) const;
    ZENO_API void set_output2(std::string const &id, zany &&obj);

    /* todo: deprecated */
    ZENO_API bool has_input(std::string const &id) const;

    /* todo: deprecated */
    ZENO_API std::shared_ptr<IObject> get_input(std::string const &id, std::string const &msg = "IObject") const;

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
        // if (!has_input2<std::shared_ptr<IObject>>(id))
        //     return false;
        auto obj = get_input(id);
        auto p = std::dynamic_pointer_cast<T>(std::move(obj));
        return (bool)p;
    }

    ZENO_API bool _implicit_cast_from_to(std::string const &id,
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
            throw Exception("input socket `" + id + "` expect IObject of `"
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
