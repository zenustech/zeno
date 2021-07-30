#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/Exception.h>
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
    std::map<std::string, std::shared_ptr<IObject>> inputs;
    std::map<std::string, std::shared_ptr<IObject>> outputs;
    std::shared_ptr<IObject> muted_output;
    std::map<std::string, IValue> params;
    std::set<std::string> options;

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
    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API IValue get_param(std::string const &id) const;
    ZENO_API std::shared_ptr<IObject> get_input(std::string const &id) const;
    ZENO_API void set_output(std::string const &id,
        std::shared_ptr<IObject> &&obj);

    template <class T>
    bool has_input(std::string const &id) const {
        if (!has_input(id))
            return false;
        auto obj = get_input(id);
        auto p = std::dynamic_pointer_cast<T>(std::move(obj));
        return (bool)p;
    }

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        auto p = std::dynamic_pointer_cast<T>(std::move(obj));
        if (!p) {
            throw Exception("input socket `" + id + "` expect `"
                    + typeid(T).name() + "`, got `"
                    + typeid(*obj.get()).name() + "`");
        }
        return p;
    }

    template <class T>
    T get_param(std::string const &id) const {
        return std::get<T>(get_param(id));
    }
};

}
