#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/types/UserData.h>
#include <functional>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>

namespace zeno {

struct Session;
struct SubgraphNode;
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
    Session *session = nullptr;
    SubgraphNode *subgraphNode = nullptr;

    std::map<std::string, std::unique_ptr<INode>> nodes;

    std::map<std::string, zany> subInputs;
    std::map<std::string, zany> subOutputs;
    std::map<std::string, std::function<zany()>> subInputPromises;

    std::set<std::string> finalOutputNodes;
    std::map<std::string, std::string> subInputNodes;
    std::map<std::string, std::string> subOutputNodes;
    std::set<std::string> subCategoryNodes;

    std::map<std::string, std::string> portalIns;
    std::map<std::string, zany> portals;

    std::unique_ptr<Context> ctx;

    bool isViewed = true;
    bool hasAnyView = false;

    //UserData userData;

    ZENO_API Graph();
    ZENO_API ~Graph();

    Graph(Graph const &) = delete;
    Graph &operator=(Graph const &) = delete;
    Graph(Graph &&) = delete;
    Graph &operator=(Graph &&) = delete;

    //ZENO_API UserData &getUserData();

    ZENO_API std::unique_ptr<INode> getOverloadNode(std::string const &id,
            std::vector<std::shared_ptr<IObject>> const &inputs) const;

    ZENO_API std::set<std::string> getGraphInputNames() const;
    ZENO_API std::set<std::string> getGraphOutputNames() const;

    ZENO_API void setGraphInputPromise(std::string const &id,
            std::function<zany()> getter);

    ZENO_API void setGraphInput(std::string const &id, zany obj);
    ZENO_API zany const &getGraphOutput(std::string const &id) const;
    ZENO_API void applyGraph();

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
    ZENO_API void setNodeInput(std::string const &id, std::string const &par,
        zany const &val);
    ZENO_API void setNodeOption(std::string const &id, std::string const &name);
    ZENO_API zany const &getNodeOutput(
        std::string const &sn, std::string const &ss) const;
    ZENO_API void loadGraph(const char *json);
    ZENO_API void setNodeParam(std::string const &id, std::string const &par,
        std::variant<int, float, std::string> const &val);  /* to be deprecated */
    ZENO_API Graph *createSubgraph(std::string const &ident);
    ZENO_API void finalizeAsSubgraph();
};

}
