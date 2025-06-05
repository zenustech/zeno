#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/funcs/LiterialConverter.h>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <zeno/types/CurveObject.h>
#include <zeno/extra/GlobalState.h>

namespace zeno {

struct Graph;
struct INodeClass;
struct Scene;
struct Session;
struct GlobalState;
struct TempNodeCaller;

struct INode {
public:
    Graph *graph = nullptr;
    INodeClass *nodeClass = nullptr;

    std::string myname;
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, zany> inputs;
    std::map<std::string, zany> outputs;
    std::set<std::string> kframes;
    std::set<std::string> formulas;
    zany muted_output;

    bool bTmpCache = false;
    std::string objRunType;

    ZENO_API INode();
    ZENO_API virtual ~INode();

    //INode(INode const &) = delete;
    //INode &operator=(INode const &) = delete;
    //INode(INode &&) = delete;
    //INode &operator=(INode &&) = delete;

    ZENO_API void doComplete();
    ZENO_API void doApply();
    ZENO_API void doOnlyApply();
    ZENO_API zany resolveInput(std::string const& id);
    ZENO_API bool getTmpCache();
    ZENO_API void writeTmpCaches();

protected:
    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;

public:
    ZENO_API bool requireInput(std::string const &ds);

    ZENO_API virtual void preApply();

    ZENO_API Graph *getThisGraph() const;
    ZENO_API Session *getThisSession() const;
    ZENO_API GlobalState *getGlobalState() const;

    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API zany get_input(std::string const &id) const;
    ZENO_API void set_output(std::string const &id, zany obj);

    ZENO_API bool has_keyframe(std::string const &id) const;
    ZENO_API zany get_keyframe(std::string const &id) const;

    ZENO_API bool has_formula(std::string const &id) const;
    ZENO_API zany get_formula(std::string const &id) const;

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        return safe_dynamic_cast<T>(std::move(obj), "input socket `" + id + "` of node `" + myname + "`");
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
    auto get_input2(std::string const &id) const {
        return objectToLiterial<T>(get_input(id), "input socket `" + id + "` of node `" + myname + "`");
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

    //[[deprecated("use get_param<T>")]]
    //ZENO_API std::variant<int, float, std::string> get_param(std::string const &id) const;

    template <class T = IObject>
    std::shared_ptr<T> get_input(std::string const &id, std::shared_ptr<T> const &defl) const {
        return has_input(id) ? get_input<T>(id) : defl;
    }

    template <class T>
    T get_input2(std::string const &id, T const &defl) const {
        return has_input(id) ? get_input2<T>(id) : defl;
    }

    ZENO_API TempNodeCaller temp_node(std::string const &id);

};

}
