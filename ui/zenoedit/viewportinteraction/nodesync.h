#ifndef VIEWPORT_NODESYNC_H
#define VIEWPORT_NODESYNC_H
#include "zenoapplication.h"

#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/graphsmanagment.h>

#include <optional>
#include <unordered_map>
namespace zeno {
struct NodeLocation{
    QModelIndex node;
    QModelIndex subgraph;
    NodeLocation(const QModelIndex& n,
                 const QModelIndex& s)
        : node(n),
          subgraph(s){
    }
    QString get_node_id() const {
        return node.data(ROLE_OBJID).toString();
    }
};

class NodeSyncMgr {
  public:
    static NodeSyncMgr &GetInstance() {
        static NodeSyncMgr instance;
        return instance;
    }
    // node generate functions
    // generate a node & link to an exist node
    std::optional<NodeLocation> generateNewNode(NodeLocation& node_location,
                                                const std::string& new_node_type,
                                                const std::string& output_sock,
                                                const std::string& input_sock);

    // node locate functions
    std::optional<NodeLocation> searchNodeOfPrim(const std::string& prim_name);
    std::optional<NodeLocation> searchNode(const std::string& node_id);

    // node check functions
    bool checkNodeType(const QModelIndex& node,
                       const std::string& node_type);
    bool checkNodeInputHasValue(const QModelIndex& node,
                                const std::string& input_name);
    std::optional<NodeLocation> checkNodeLinkedSpecificNode(const QModelIndex& node,       // check which node's output?
                                                            const std::string& node_type); // check node output linked which node type
    // get input or output
    std::vector<NodeLocation> getInputNodes(const QModelIndex& node,
                                            const std::string& input_name);
    std::string getInputValString(const QModelIndex& node,
                                  const std::string& input_name);
    std::string getParamValString(const QModelIndex& node,
                                  const std::string& param_name);

    // node update functions
    void updateNodeVisibility(NodeLocation& node_location);
    template <class T>
    void updateNodeInputVec(NodeLocation& node_location,
                            const std::string& input_name,
                            const QVector<T>& new_value) {
        auto node_id = node_location.node.data(ROLE_OBJID).toString();
        auto inputs = node_location.node.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        auto old_value = inputs[input_name.c_str()].info.defaultValue.value<UI_VECTYPE>();
        PARAM_UPDATE_INFO update_info{
            input_name.c_str(),
            QVariant::fromValue(old_value),
            QVariant::fromValue(new_value)
        };
        m_graph_model->updateSocketDefl(node_id,
                                        update_info,
                                        node_location.subgraph,
                                        true);
    }
    void updateNodeInputString(NodeLocation node_location,
                               const std::string& input_name,
                               const std::string& new_value);
    void updateNodeParamString(NodeLocation node_location,
                               const std::string& param_name,
                               const std::string& new_value);

    // other tool functions
    std::string getPrimSockName(const std::string& node_type);
    std::string getPrimSockName(NodeLocation& node_location);

    NodeSyncMgr(const NodeSyncMgr &) = delete;
    const NodeSyncMgr &operator=(const NodeSyncMgr &) = delete;

  private:
    IGraphsModel *m_graph_model;    //bug: the currentmodel will be invalidated after the current graph has been closed.
    std::unordered_map<std::string, std::string> m_prim_sock_map;
    NodeSyncMgr() {
        m_graph_model = zenoApp->graphsManagment()->currentModel();
        registerDefaultSocketName();
    };
    void registerDefaultSocketName() {
        m_prim_sock_map["BindMaterial"] = "object";
        m_prim_sock_map["TransformPrimitive"] = "outPrim";
    }
};

}

#endif //VIEWPORT_NODESYNC_H
