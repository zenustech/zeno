#pragma once

#include <zeno/utils/defs.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/any.h>
#include <memory>
#include <string>
#include <set>
#include <map>

namespace zeno {

struct Scene;
struct INode;

struct Context {
    std::set<std::string> visited;

    inline void mergeVisited(Context const &other) {
        visited.insert(other.visited.begin(), other.visited.end());
    }

    ZENO_API Context();
    ZENO_API Context(Context const &other);
    ZENO_API ~Context();
};

struct Graph {
    Scene *scene = nullptr;

    std::map<std::string, std::unique_ptr<INode>> nodes;

    std::map<std::string, any> subInputs;
    std::map<std::string, any> subOutputs;
    std::map<std::string, std::string> subOutputNodes;

    std::map<std::string, std::string> portalIns;
    std::map<std::string, any> portals;

    std::unique_ptr<Context> ctx;

    bool isViewed = true;
    bool hasAnyView = false;

    ZENO_API Graph();
    ZENO_API ~Graph();

    ZENO_API void setGraphInput2(std::string const &id, any obj);
    ZENO_API any const &getGraphOutput2(std::string const &id) const;
    ZENO_API void applyGraph();

    void setGraphInput(std::string const &id,
            std::shared_ptr<IObject> obj) {
        setGraphInput2(id, std::move(obj));
    }
    std::shared_ptr<IObject> getGraphOutput(
            std::string const &id) const {
        return smart_any_cast<std::shared_ptr<IObject>>(getGraphOutput2(id));
    }

    template <class T>
    std::shared_ptr<T> getGraphOutput(
            std::string const &id) const {
        auto obj = getGraphOutput(id);
        return safe_dynamic_cast<T>(std::move(obj),
                "graph output `" + id + "` ");
    }

    ZENO_API void clearNodes();
    ZENO_API void applyNodes(std::set<std::string> const &ids);
    ZENO_API void addNode(std::string const &cls, std::string const &id);
    ZENO_API void applyNode(std::string const &id);
    ZENO_API void completeNode(std::string const &id);
    ZENO_API void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);
    ZENO_API void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val);
    ZENO_API void setNodeOption(std::string const &id, std::string const &name);
    ZENO_API any const &getNodeOutput(
        std::string const &sn, std::string const &ss) const;
};

}
