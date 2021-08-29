#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/any.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>

namespace zeno {

struct Graph;
struct INodeClass;

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

    ZENO_API void doComplete();
    ZENO_API virtual void doApply();

protected:
    ZENO_API bool checkApplyCondition();
    ZENO_API void requireInput(std::string const &ds);
    ZENO_API void coreApply();

    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;

    ZENO_API bool has_option(std::string const &id) const;
    ZENO_API bool has_input2(std::string const &id) const;
    ZENO_API zany get_input2(std::string const &id) const;
    ZENO_API void set_output2(std::string const &id, zany &&obj);

    /* deprecated */
    bool has_input(std::string const &id) const {
        return inputBounds.find(id) != inputBounds.end();
    }

    /* deprecated */
    void set_output(std::string const &id, std::shared_ptr<IObject> &&obj) {
        set_output2(id, std::move(obj));
    }

    /* deprecated */
    std::shared_ptr<IObject> get_input(std::string const &id) const {
        return get_input2<std::shared_ptr<IObject>>(id);
    }

    template <class T>
    T get_input2(std::string const &id) const {
        return smart_any_cast<T>(get_input2(id), "input `" + id + "` ");
    }

    template <class T>
    bool has_input2(std::string const &id) const {
        if (!has_input2(id))
            return false;
        return silent_any_cast<T>(get_input2(id)).has_value();
    }

    /* deprecated */
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

    /* deprecated */
    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        return safe_dynamic_cast<T>(std::move(obj),
                "input socket `" + id + "` ");
    }

    /* deprecated */
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

    /* deprecated */
    template <class T>
    T get_param(std::string const &id) const {
        //return std::get<T>(get_param(id));
        return get_input2<T>(id + ":");
    }
};

}
