#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/Exception.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/funcs/LiterialConverter.h>
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
    //std::set<std::string> options;
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
    //ZENO_API bool checkApplyCondition();
    ZENO_API bool requireInput(std::string const &ds);

    ZENO_API virtual void preApply();
    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;

    ZENO_API Graph *getThisGraph() const;
    ZENO_API Session *getThisSession() const;
    ZENO_API GlobalState *getGlobalState() const;

    //ZENO_API bool has_option(std::string const &id) const;
    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API zany get_input(std::string const &id) const;
    ZENO_API void set_output(std::string const &id, zany obj);

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        return safe_dynamic_cast<T>(std::move(obj), "input socket `" + id + "` ");
    }

    template <class T>
    bool has_input(std::string const &id) const {
        if (!has_input(id)) return false;
        auto obj = get_input(id);
        return !!dynamic_cast<T *>(obj.get());
    }

    template <class T>
    bool has_input2(std::string const &id) const {
        if (!has_input(id)) return false;
        return objectIsLiterial<T>(get_input(id));
    }

    template <class T>
    T get_input2(std::string const &id) const {
        return objectToLiterial<T>(get_input(id));
    }

    template <class T>
    void set_output2(std::string const &id, T &&value) {
        set_output(id, objectFromLiterial(std::forward<T>(value)));
    }

    template <class T>
    [[deprecated("use get_input2<T>(id + ':')")]]
    T get_param(std::string const &id) const {
        return get_input2<T>(id + ':');
    }

    template <int = 0>
    [[deprecated("use get_param<T>")]]
    std::variant<int, float, std::string> get_param(std::string const &id) const {
        auto nid = id + ':';
        if (has_input2<int>(nid)) {
            return get_input2<int>(nid);
        }
        if (has_input2<float>(nid)) {
            return get_input2<float>(nid);
        }
        if (has_input2<std::string>(nid)) {
            return get_input2<std::string>(nid);
        }
        throw Exception("bad get_param (variant mode)");
    }
};

}
