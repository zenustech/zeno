#include "nodesync.h"
#include <zenomodel/include/uihelper.h>


namespace zeno {
std::optional<NodeLocation> NodeSyncMgr::generateNewNode(NodeLocation& node_location,
                                                         const std::string& new_node_type,
                                                         const std::string& output_sock,
                                                         const std::string& input_sock) {
    auto& node = node_location.node;
    auto& subgraph = node_location.subgraph;
    auto pos = node.data(ROLE_OBJPOS).toPointF();
    pos.setX(pos.x() + 10);
    auto new_node_id = NodesMgr::createNewNode(m_graph_model,
                                               subgraph,
                                               new_node_type.c_str(),
                                               pos);
    auto this_node_id = node.data(ROLE_OBJID).toString();

    const QString& subgName = subgraph.data(ROLE_OBJNAME).toString();
    const QString& outNode = this_node_id;
    const QString& inNode = new_node_id;
    const QString& outSock = QString::fromLocal8Bit(output_sock.c_str());
    const QString& inSock = QString::fromLocal8Bit(input_sock.c_str());

    QString outSockObj = UiHelper::constructObjPath(subgName, outNode, "[node]/outputs/", outSock);
    QString inSockObj = UiHelper::constructObjPath(subgName, inNode, "[node]/inputs/", inSock);

    EdgeInfo edge(outSockObj, inSockObj);
    m_graph_model->addLink(edge, false);
    return searchNode(new_node_id.toStdString());
}

std::optional<NodeLocation> NodeSyncMgr::searchNodeOfPrim(const std::string& prim_name) {
    QString node_id(prim_name.substr(0, prim_name.find_first_of(':')).c_str());
    return searchNode(node_id.toStdString());
}

std::optional<NodeLocation> NodeSyncMgr::searchNode(const std::string& node_id) {
    auto search_result = m_graph_model->search(node_id.c_str(),
                                               SEARCH_NODEID);
    if (search_result.empty()) return {};
    return NodeLocation(search_result[0].targetIdx,
                        search_result[0].subgIdx);
}

bool NodeSyncMgr::checkNodeType(const QModelIndex& node,
                                const std::string& node_type) {
    auto node_id = node.data(ROLE_OBJID).toString();
    return node_id.contains(node_type.c_str());
}

bool NodeSyncMgr::checkNodeInputHasValue(const QModelIndex& node,
                                         const std::string& input_name) {
    auto inputs = node.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    QString inSock = QString::fromLocal8Bit(input_name.c_str());
    if (inputs.find(inSock) == inputs.end())
        return false;

    const INPUT_SOCKET& inSocket = inputs[inSock];
    return inSocket.info.links.isEmpty();
}

std::optional<NodeLocation> NodeSyncMgr::checkNodeLinkedSpecificNode(const QModelIndex& node,
                                                                     const std::string& node_type) {
    auto this_outputs = node.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    auto this_node_id = node.data(ROLE_OBJID).toString(); // TransformPrimitive-1f4erf21
    auto this_node_type = this_node_id.section("-", 1); // TransformPrimitive
    auto prim_sock_name = getPrimSockName(this_node_type.toStdString());

    QString sockName = QString::fromLocal8Bit(prim_sock_name.c_str());
    if (this_outputs.find(sockName) == this_outputs.end())
        return {};

    auto linked_edges = this_outputs[sockName].info.links;
    for (const auto& linked_edge : linked_edges) {
        auto next_node_id = UiHelper::getSockNode(linked_edge.inSockPath);
        if (next_node_id.contains(node_type.c_str())) {
            auto search_result = m_graph_model->search(next_node_id,
                                                       SEARCH_NODEID);
            if (search_result.empty()) return {};
            auto linked_node = search_result[0].targetIdx;
            auto linked_subgraph = search_result[0].subgIdx;
            auto option = linked_node.data(ROLE_OPTIONS).toInt();
            if (option & OPT_VIEW)
                return NodeLocation(linked_node,
                                    linked_subgraph);
        }
    }
    return {};
}

std::vector<NodeLocation> NodeSyncMgr::getInputNodes(const QModelIndex& node,
                                                     const std::string& input_name) {
    std::vector<NodeLocation> res;
    auto inputs = node.data(ROLE_INPUTS).value<INPUT_SOCKETS>();

    QString sockName = QString::fromLocal8Bit(input_name.c_str());
    if (inputs.find(sockName) == inputs.end())
        return res;

    for (const auto& input_edge : inputs[sockName].info.links) {
        auto input_node_id = UiHelper::getSockName(input_edge.outSockPath);
        auto searched_node = searchNode(input_node_id.toStdString());
        if (searched_node.has_value())
            res.emplace_back(searched_node.value());
    }
    return res;
}

std::string NodeSyncMgr::getInputValString(const QModelIndex& node,
                                           const std::string& input_name) {
    auto inputs = node.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    return inputs[input_name.c_str()].info.defaultValue.value<QString>().toStdString();
}

std::string NodeSyncMgr::getParamValString(const QModelIndex& node,
                                           const std::string& param_name) {
    auto params = node.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    return params.value(param_name.c_str()).value.value<QString>().toStdString();
}

void NodeSyncMgr::updateNodeVisibility(NodeLocation& node_location) {
    auto node_id = node_location.node.data(ROLE_OBJID).toString();
    int old_option = node_location.node.data(ROLE_OPTIONS).toInt();
    int new_option = old_option;
    new_option ^= OPT_VIEW;
    STATUS_UPDATE_INFO status_info = {old_option, new_option, ROLE_OPTIONS};
    m_graph_model->updateNodeStatus(node_id,
                                    status_info,
                                    node_location.subgraph,
                                    true);
}

void NodeSyncMgr::updateNodeInputString(NodeLocation node_location,
                                        const std::string& input_name,
                                        const std::string& new_value) {
    auto node_id = node_location.node.data(ROLE_OBJID).toString();
    auto inputs = node_location.node.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    auto old_value = inputs[input_name.c_str()].info.defaultValue.value<QString>();
    PARAM_UPDATE_INFO update_info{
        input_name.c_str(),
        QVariant::fromValue(old_value),
        QVariant::fromValue(QString(new_value.c_str()))
    };
    m_graph_model->updateSocketDefl(node_id,
                                    update_info,
                                    node_location.subgraph,
                                    true);
}

void NodeSyncMgr::updateNodeParamString(NodeLocation node_location,
                                        const std::string& param_name,
                                        const std::string& new_value) {
    auto params = node_location.node.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    PARAM_INFO info = params.value(param_name.c_str());
    PARAM_UPDATE_INFO new_info = {
        param_name.c_str(),
        info.value,
        QVariant(new_value.c_str())
    };

    m_graph_model->updateParamInfo(node_location.get_node_id(),
                                   new_info,
                                   node_location.subgraph,
                                   true);
}

std::string NodeSyncMgr::getPrimSockName(const std::string& node_type) {
    if (m_prim_sock_map.find(node_type) != m_prim_sock_map.end())
        return m_prim_sock_map[node_type];
    return "prim";
}

std::string NodeSyncMgr::getPrimSockName(NodeLocation& node_location) {
    auto node_type = node_location.node.data(ROLE_OBJID).toString().section("-", 1);
    return getPrimSockName(node_type.toStdString());
}

}