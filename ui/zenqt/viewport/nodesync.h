#ifndef VIEWPORT_NODESYNC_H
#define VIEWPORT_NODESYNC_H

#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "util/curveutil.h"
#include <optional>
#include <unordered_map>
#include "widgets/ztimeline.h"
#include "zassert.h"


namespace zeno {
struct NodeLocation{
    QModelIndex node;
    GraphModel* subgraph;
    NodeLocation(const QModelIndex& n,
                 GraphModel* s)
        : node(n),
          subgraph(s){
    }
    QString get_node_id() const {
        return node.data(ROLE_NODE_NAME).toString();
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
    bool getNewValue(const QVariant & old_value, const QVector<T>& new_value, QVariant &value)
    {
        if (old_value.canConvert<CURVES_DATA>())
        {
            auto curves = old_value.value<CURVES_DATA>();
            bool bUpdate = false;
            for (int idx = 0; idx < new_value.size(); idx++)
            {
                QString key = curve_util::getCurveKey(idx);
                if (curves.contains(key))
                {
                    auto& curve = curves[key];
                    auto pMainWin = zenoApp->getMainWindow();
                    ZASSERT_EXIT(pMainWin, false);
                    auto pTimeline = pMainWin->timeline();
                    ZASSERT_EXIT(pTimeline, false);
                    int x = pTimeline->value();
                    if (curve_util::updateCurve(QPointF(x, new_value[idx]), curve))
                        bUpdate = true;
                }
            }
            if (!bUpdate)
                return false;
            value = QVariant::fromValue(curves);
        }
        else if (old_value.canConvert<UI_VECTYPE>())
        {
            value = QVariant::fromValue(new_value);
        }
        else
        {
            return false;
        }
        return true;
    }

    template <class T>
    void updateNodeInputVec(NodeLocation& node_location,
                            const std::string& input_name,
                            const QVector<T>& new_value) {
        auto graph_model = zenoApp->graphsManager()->currentModel();
        if (!graph_model) {
            return;
        }

        auto node_id = node_location.node.data(ROLE_NODE_NAME).toString();
        auto inputs = node_location.node.data(ROLE_INPUTS).value<PARAMS_INFO>();
        //TODO: zvariant to qvariant.
        QVariant old_value;// = inputs[input_name.c_str()].defl;
        QVariant value;
        if (!getNewValue<T>(old_value, new_value, value))
            return;
        //PARAM_UPDATE_INFO update_info{
        //    input_name.c_str(),
        //    old_value,
        //    value
        //};
        //graph_model->updateSocketDefl(node_id,
        //                              update_info,
        //                              node_location.subgraph,
        //                              true);
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
    std::unordered_map<std::string, std::string> m_prim_sock_map;
    NodeSyncMgr() {
        registerDefaultSocketName();
    };
    void registerDefaultSocketName() {
        m_prim_sock_map["BindMaterial"] = "object";
        m_prim_sock_map["TransformPrimitive"] = "outPrim";
    }
};

}

#endif //VIEWPORT_NODESYNC_H
