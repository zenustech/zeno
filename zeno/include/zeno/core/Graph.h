#pragma once

#include "IObject.h"

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

    std::map<std::string, std::shared_ptr<IObject>> subInputs;
    std::map<std::string, std::shared_ptr<IObject>> subOutputs;
    std::map<std::string, std::string> subOutputNodes;

    std::map<std::string, std::string> portalIns;
    std::map<std::string, std::shared_ptr<IObject>> portals;

    std::unique_ptr<Context> ctx;

    bool isViewed = true;
    bool hasAnyView = false;

    ZENO_API Graph();
    ZENO_API ~Graph();

    ZENO_API void setGraphInput(std::string const &id,
            std::shared_ptr<IObject> obj);
    ZENO_API std::shared_ptr<IObject> getGraphOutput(
            std::string const &id) const;
    ZENO_API void applyGraph();

    template <class T>
    std::shared_ptr<T> getGraphOutput(
            std::string const &id) const {
        auto obj = getGraphOutput(id);
        auto p = std::dynamic_pointer_cast<T>(obj);
        if (!p) {
            throw Exception("graph output `" + id + "` expect `"
                    + typeid(T).name() + "`, got `"
                    + typeid(*obj.get()).name() + "`");
        }
        return p;
    }

    ZENO_API void clearNodes();
    ZENO_API void applyNodes(std::vector<std::string> const &ids);
    ZENO_API void addNode(std::string const &cls, std::string const &id);
    ZENO_API void applyNode(std::string const &id);
    ZENO_API void completeNode(std::string const &id);
    ZENO_API void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);
    ZENO_API void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val);
    ZENO_API void setNodeOption(std::string const &id, std::string const &name);
    ZENO_API std::shared_ptr<IObject> const &getNodeOutput(
        std::string const &sn, std::string const &ss) const;
};

struct Scene {
    std::map<std::string, std::unique_ptr<Graph>> graphs;

    Graph *currGraph = nullptr;
    Session *sess = nullptr;

    ZENO_API Scene();
    ZENO_API ~Scene();

    ZENO_API Graph &getGraph();
    ZENO_API void clearAllState();
    ZENO_API Graph &getGraph(std::string const &name) const;
    ZENO_API void switchGraph(std::string const &name);
    ZENO_API void loadScene(const char *json);
};
