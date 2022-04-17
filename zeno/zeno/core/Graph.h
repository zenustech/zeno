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
#include <any>
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
    std::map<std::string, std::string> portalIns;
    std::map<std::string, zany> portals;
    std::set<std::string> nodesToExec;
    int adhocNumFrames = 0;

    std::unique_ptr<Context> ctx;

    ZENO_API Graph();
    ZENO_API ~Graph();

    Graph(Graph const &) = delete;
    Graph &operator=(Graph const &) = delete;
    Graph(Graph &&) = delete;
    Graph &operator=(Graph &&) = delete;

    ZENO_API void clearNodes();
    ZENO_API void applyNodesToExec();
    ZENO_API void applyNodes(std::set<std::string> const &ids);
    ZENO_API void addNode(std::string const &cls, std::string const &id);
    ZENO_API void applyNode(std::string const &id);
    ZENO_API void completeNode(std::string const &id);
    ZENO_API void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);
    ZENO_API void setNodeInput(std::string const &id, std::string const &par,
        zany const &val);
    ZENO_API zany const &getNodeOutput(std::string const &sn, std::string const &ss) const;
    ZENO_API void loadGraph(const char *json);
    ZENO_API void setNodeParam(std::string const &id, std::string const &par,
        std::variant<int, float, std::string> const &val);  /* to be deprecated */
    ZENO_API std::unique_ptr<INode> getOverloadNode(std::string const &id,
            std::vector<std::shared_ptr<IObject>> const &inputs) const;
};

}
