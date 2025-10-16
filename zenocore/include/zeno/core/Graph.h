#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/core/data.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <functional>
#include <variant>
#include <memory>
#include <unordered_map>
#include <string>
#include <set>
#include <map>
#include <zeno/core/data.h>
#include <zeno/utils/uuid.h>


namespace zeno {

struct Session;
struct SubgraphNode;
struct SubnetNode;
struct DirtyChecker;
struct NodeImpl;

struct Context {
    std::set<std::string> visited;

    inline void mergeVisited(Context const &other) {
        visited.insert(other.visited.begin(), other.visited.end());
    }

    Context();
    Context(Context const &other);
    ~Context();
};

struct ZENO_API Graph : public std::enable_shared_from_this<Graph> {
    Session* session = nullptr;

    Graph(const std::string& name, bool bAssets = false);
    ~Graph();

    Graph(Graph const &) = delete;
    Graph &operator=(Graph const &) = delete;
    Graph(Graph &&) = delete;
    Graph &operator=(Graph &&) = delete;

    //BEGIN NEW STANDARD API
    void init(const GraphData& graph);
    void initRef(const GraphData& graph);

    NodeImpl* createNode(
        const std::string& cls,
        const std::string& orgin_name = "",
        bool bAssets = false,
        std::pair<float, float> pos = {},
        bool isIOInit = false,
        bool* pbAssetLock = nullptr);
    CALLBACK_REGIST(createNode, void, const std::string&, zeno::NodeImpl*)

    bool removeNode(std::string const& name);
    CALLBACK_REGIST(removeNode, void, const std::string&)

    bool addLink(const EdgeInfo& edge);
    CALLBACK_REGIST(addLink, bool, EdgeInfo)

    bool removeLink(const EdgeInfo& edge);
    CALLBACK_REGIST(removeLink, bool, EdgeInfo)

    bool removeLinks(const std::string nodename, bool bInput, const std::string paramname);
    CALLBACK_REGIST(removeLinks, bool, std::string, bool, std::string)

    void update_load_info(const std::string& nodecls, bool bDisable);

    bool updateLink(const EdgeInfo& edge, bool bInput, const std::string oldkey, const std::string newkey);
    bool moveUpLinkKey(const EdgeInfo& edge, bool bInput, const std::string keyName);

    bool hasNode(std::string const& uuid_node_path);
    //老代码直接拿NodeImpl就行了，不需要考虑abi问题
    NodeImpl* getNode(std::string const& name);
    NodeImpl* getNodeByUuidPath(ObjPath path);
    NodeImpl* getNodeByPath(const std::string& path);
    NodeImpl* getParentSubnetNode() const;
    void initParentSubnetNode(NodeImpl* pSubnetNode);
    std::vector<NodeImpl*> getNodesByClass(const std::string& cls);
    std::shared_ptr<Graph> getGraphByPath(const std::string& path);
    std::map<std::string, NodeImpl*> getNodes() const;
    std::set<std::string> get_viewnodes() const;
    float statistic_cpu_used() const;

    GraphData exportGraph() const;

    LinksData exportLinks() const;

    std::string getName() const;
    void setName(const std::string& name);

    std::string updateNodeName(const std::string oldName, const std::string newName = "");
    CALLBACK_REGIST(updateNodeName, void, std::string, std::string)

    void clear();
    CALLBACK_REGIST(clear, void)

    bool isAssets() const;
    bool isAssetRoot() const;
    std::set<std::string> searchByClass(const std::string& name) const;

    void clearNodes();
    void clearContainerUpdateInfo();
    void runGraph(render_reload_info& infos);
    void mark_clean();
    void applyNodes(std::set<std::string> const &ids, render_reload_info& infos);
    void addNode(std::string const &cls, std::string const &id);
    Graph *addSubnetNode(std::string const &id);
    Graph *getSubnetGraph(std::string const &id) const;
    void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);

    //容易有歧义：这个input是defl value，还是实质的对象？按原来zeno的语义，是指defl value
    void setNodeInput(std::string const &id, std::string const &par, zany const &val);

    void setKeyFrame(std::string const &id, std::string const &par, zany const &val);
    void setFormula(std::string const &id, std::string const &par, zany const &val);
    void addNodeOutput(std::string const &id, std::string const &par);
    std::map<std::string, zany> callSubnetNode(std::string const &id,
            std::map<std::string, zany> inputs) const;

    std::set<std::string> getSubInputs();
    std::set<std::string> getSubOutputs();
    void viewNodeUpdated(const std::string node, bool bView);
    void markDirtyWhenFrameChanged();
    void markDirtyAndCleanup();
    void onNodeParamUpdated(PrimitiveParam* spParam, zeno::reflect::Any old_value, zeno::reflect::Any new_value);
    void parseNodeParamDependency(PrimitiveParam* spParam, zeno::reflect::Any& new_value);

    bool isFrameNode(std::string uuid);

    //增/删边之后更新wildCard端口的类型
    void updateWildCardParamTypeRecursive(Graph* spCurrGarph, NodeImpl* spNode, std::string paramName, bool bPrim, bool bInput, ParamType newtype);

private:
    std::string generateNewName(const std::string& node_cls, const std::string& origin_name = "", bool bAssets = false);
    //增/删边之后更新wildCard端口的类型
    void removeLinkWhenUpdateWildCardParam(const std::string& outNode, const std::string& inNode, EdgeInfo& edge);
    void resetWildCardParamsType(bool bWildcard, NodeImpl* node, const std::string& paramName, const bool& bPrimType, const bool& bInput);
    std::shared_ptr<Graph> _getGraphByPath(std::vector<std::string> items);
    bool isLinkValid(const EdgeInfo& edge);
    render_update_info applyNode(std::string const& node_name, CalcContext* pContext);

    NodeImpl* m_parSubnetNode = nullptr;
    std::map<std::string, std::unique_ptr<NodeImpl>> m_nodes;  //based on uuid.
    std::set<std::string> nodesToExec;
    std::map<std::string, std::string> subInputNodes;
    std::map<std::string, std::string> subOutputNodes;
    std::map<std::string, std::string> m_name2uuid;
    std::map<std::string, std::set<std::string>> node_set;  //cls to name
    std::set<std::string> frame_nodes;      //record all nodes depended on frame num.
    std::set<std::string> subnet_nodes;     //base uuid
    std::set<std::string> asset_nodes;
    std::set<std::string> subinput_nodes;
    std::set<std::string> suboutput_nodes;
    std::set<std::string> m_viewnodes;
    std::string m_name;

    const bool m_bAssets;   //只是说明这个图是一个资产，可以是实例也可以是资产图引用
};

}
