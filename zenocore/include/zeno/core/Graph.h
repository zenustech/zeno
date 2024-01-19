#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/core/INode.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/types/UserData.h>
#include <functional>
#include <variant>
#include <memory>
#include <unordered_map>
#include <string>
#include <set>
#include <any>
#include <map>
#include <zeno/core/data.h>
#include <zeno/utils/uuid.h>

namespace zeno {

struct Session;
struct SubgraphNode;
struct DirtyChecker;
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

struct Graph : std::enable_shared_from_this<Graph> {
    //SubgraphNode *subgraphNode = nullptr;

    std::map<std::string, std::shared_ptr<INode>> nodes;
    std::set<std::string> nodesToExec;
    int beginFrameNumber = 0, endFrameNumber = 0;  // only use by runnermain.cpp

    std::map<std::string, std::string> portalIns;   //todo: deprecated, but need to keep compatible with old zsg.
    std::map<std::string, zany> portals;

    std::unique_ptr<Context> ctx;
    std::unique_ptr<DirtyChecker> dirtyChecker;

    ZENO_API Graph(const std::string& name);
    ZENO_API ~Graph();

    Graph(Graph const &) = delete;
    Graph &operator=(Graph const &) = delete;
    Graph(Graph &&) = delete;
    Graph &operator=(Graph &&) = delete;

    //BEGIN NEW STANDARD API
    ZENO_API void init(const GraphData& graph);

    ZENO_API std::shared_ptr<INode> createNode(std::string const& cls);
    CALLBACK_REGIST(createNode, void, const std::string&, std::weak_ptr<zeno::INode>)

    ZENO_API bool removeNode(std::string const& name);
    CALLBACK_REGIST(removeNode, void, const std::string&)

    ZENO_API std::shared_ptr<INode> createSubnetNode(std::string const& name);
    CALLBACK_REGIST(createSubnetNode, void, std::weak_ptr<zeno::INode>)

    ZENO_API bool addLink(const EdgeInfo& edge);
    CALLBACK_REGIST(addLink, bool, EdgeInfo)

    ZENO_API bool removeLink(const EdgeInfo& edge);
    CALLBACK_REGIST(removeLink, bool, EdgeInfo)

    ZENO_API std::shared_ptr<INode> getNode(std::string const& name);
    ZENO_API std::map<std::string, std::shared_ptr<INode>> getNodes() const;
    ZENO_API std::string getName() const;
    //END

    ZENO_API DirtyChecker &getDirtyChecker();
    ZENO_API void clearNodes();
    ZENO_API void applyNodesToExec();
    ZENO_API void applyNodes(std::set<std::string> const &ids);
    ZENO_API void addNode(std::string const &cls, std::string const &id);
    ZENO_API Graph *addSubnetNode(std::string const &id);
    ZENO_API Graph *getSubnetGraph(std::string const &id) const;
    ZENO_API bool applyNode(std::string const &id);
    ZENO_API void completeNode(std::string const &id);
    ZENO_API void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);

    //容易有歧义：这个input是defl value，还是实质的对象？按原来zeno的语义，是指defl value
    ZENO_API void setNodeInput(std::string const &id, std::string const &par,
        zany const &val);

    ZENO_API void setKeyFrame(std::string const &id, std::string const &par, zany const &val);
    ZENO_API void setFormula(std::string const &id, std::string const &par, zany const &val);
    ZENO_API void addNodeOutput(std::string const &id, std::string const &par);
    ZENO_API zany getNodeInput(std::string const &sn, std::string const &ss) const;
    ZENO_API void loadGraph(const char *json);
    ZENO_API void setNodeParam(std::string const &id, std::string const &par,
        std::variant<int, float, std::string, zany> const &val);  /* to be deprecated */
    ZENO_API std::map<std::string, zany> callSubnetNode(std::string const &id,
            std::map<std::string, zany> inputs) const;
    ZENO_API std::map<std::string, zany> callTempNode(std::string const &id,
            std::map<std::string, zany> inputs) const;

    std::map<std::string, std::string> getSubInputs();
    std::map<std::string, std::string> getSubOutputs();

private:
    std::string generateNewName(const std::string& node_cls);

    std::map<std::string, std::string> subInputNodes;
    std::map<std::string, std::string> subOutputNodes;

    std::map<std::string, std::set<std::string>> node_set;
    std::string name;
};

}
