#include "nodesync.h"
#include "util/uihelper.h"
#include "uicommon.h"
#include "model/graphsmanager.h"
#include "model/graphstreemodel.h"
#include "zassert.h"
#include <zeno/core/ObjectManager.h>
#include <zeno/core/Session.h>
#include "reflect/reflection.generated.hpp"


namespace zeno {
std::optional<NodeLocation> NodeSyncMgr::generateNewNode(NodeLocation& node_location,
                                                         const std::string& new_node_type,
                                                         const std::string& output_sock,
                                                         const std::string& input_sock)
{
    QModelIndex nodeIdx = node_location.node;
    if (!nodeIdx.isValid()) {
        return {};
    }

    auto subgraph = node_location.subgraph;
    auto pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
    pos.setX(pos.x() + 100);

    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(nodeIdx.model());
    GraphModel* pGraphM = qobject_cast<GraphModel*>(pModel);
    ZASSERT_EXIT(pGraphM, {});

    auto new_node_id = UiHelper::createNewNode(subgraph, QString::fromStdString(new_node_type), pos);

    auto this_node_id = nodeIdx.data(ROLE_NODE_NAME).toString();

    const QString& subgName = subgraph->name();
    const QString& outNode = this_node_id;
    const QString& inNode = new_node_id;
    const QString& outSock = QString::fromLocal8Bit(output_sock.c_str());
    const QString& inSock = QString::fromLocal8Bit(input_sock.c_str());

    QString outSockObj = UiHelper::constructObjPath(subgName, outNode, "[node]/outputs/", outSock);
    QString inSockObj = UiHelper::constructObjPath(subgName, inNode, "[node]/inputs/", inSock);

    zeno::EdgeInfo edge = {
        outNode.toStdString(),
        outSock.toStdString(),
        "",
        inNode.toStdString(),
        inSock.toStdString(),
        ""};

    pGraphM->addLink(edge);
    return searchNode(new_node_id.toStdString());
}

std::optional<NodeLocation> NodeSyncMgr::searchNodeOfPrim(const std::string& prim_name) {
    auto graph_model = zenoApp->graphsManager()->currentModel();
    if (!graph_model) {
        return {};
    }

    const auto& objsMan = zeno::getSession().objsMan;
    std::set<ObjPath> nodeNameSet = objsMan->getAttachNodes(prim_name);
    if (nodeNameSet.empty())
        return {};

    for (auto nodepath : nodeNameSet)
    {
        auto spNode = zeno::getSession().mainGraph->getNodeByUuidPath(nodepath);
        if (spNode && spNode->is_view()) {
            auto search_result = graph_model->searchByUuidPath(*nodeNameSet.begin());
            return NodeLocation(search_result[0].targetIdx, search_result[0].subGraph);
        }
    }
    return {};
}

std::optional<NodeLocation> NodeSyncMgr::searchNode(const std::string& node_id) {
    auto graph_model = zenoApp->graphsManager()->currentModel();
    if (!graph_model) {
        return {};
    }
    auto search_result = graph_model->search(QString::fromStdString(node_id), SEARCH_NODEID, SEARCH_MATCH_EXACTLY);
    if (search_result.empty())
        return {};
    return NodeLocation(search_result[0].targetIdx,
                        search_result[0].subGraph);
}

bool NodeSyncMgr::checkNodeType(const QModelIndex& node,
                                const std::string& node_type) {
    auto node_id = node.data(ROLE_NODE_NAME).toString();
    return node_id.contains(node_type.c_str());
}

bool NodeSyncMgr::checkNodeInputHasValue(const QModelIndex& node, const std::string& input_name)
{
    auto inputs = node.data(ROLE_INPUTS).value<PARAMS_INFO>();
    QString inSock = QString::fromLocal8Bit(input_name.c_str());
    if (inputs.find(inSock) == inputs.end())
        return false;

    const auto& inSocket = inputs[inSock];
    return inSocket.links.empty();
}

std::optional<NodeLocation> NodeSyncMgr::checkNodeLinkedSpecificNode(NodeLocation& node_location,
                                                                     const std::string& node_type) {
    auto graph_model = zenoApp->graphsManager()->currentModel();
    if (!graph_model) {
        return {};
    }

    auto this_outputs = node_location.node.data(ROLE_OUTPUTS).value<PARAMS_INFO>();
    auto this_node_id = node_location.node.data(ROLE_NODE_NAME).toString(); // BindMaterial1
    auto prim_sock_name = getPrimSockName(node_location);

    QString sockName = QString::fromLocal8Bit(prim_sock_name.c_str());
    if (this_outputs.find(sockName) == this_outputs.end())
        return {};

    auto linked_edges = this_outputs[sockName].links;
    for (const auto& linked_edge : linked_edges) {
        auto next_node_id = QString::fromStdString(linked_edge.inNode);
        if (next_node_id.contains(node_type.c_str())) {
            auto search_result = graph_model->search(next_node_id, SEARCH_NODEID, SEARCH_MATCH_EXACTLY);
            if (search_result.empty()) return {};
            auto linked_node = search_result[0].targetIdx;
            auto linked_subgraph = search_result[0].subGraph;
            auto option = linked_node.data(ROLE_NODE_STATUS).toInt();
            if (option & zeno::View)
                return NodeLocation(linked_node,
                    linked_subgraph);
        }
    }
    return {};
}

std::vector<NodeLocation> NodeSyncMgr::getInputNodes(const QModelIndex& node,
                                                     const std::string& input_name) {
    std::vector<NodeLocation> res;
    auto inputs = node.data(ROLE_INPUTS).value<PARAMS_INFO>();

    QString sockName = QString::fromLocal8Bit(input_name.c_str());
    if (inputs.find(sockName) == inputs.end())
        return res;

    for (const auto& input_edge : inputs[sockName].links) {
        auto output_node_id = input_edge.outNode;
        auto searched_node = searchNode(output_node_id);
        if (searched_node.has_value())
            res.emplace_back(searched_node.value());
    }
    return res;
}

std::string NodeSyncMgr::getInputValString(const QModelIndex& node,
                                           const std::string& input_name)
{
    auto inputs = node.data(ROLE_INPUTS).value<PARAMS_INFO>();
    const QString& paramName = QString::fromStdString(input_name);
    std::string ret;
    if (inputs.find(paramName) != inputs.end()) {
        auto& param = inputs[paramName];
        zeno::zeno_get_if(param.defl, ret);
    }
    return ret;
}

std::string NodeSyncMgr::getParamValString(const QModelIndex& node,
                                           const std::string& param_name) {
    return getInputValString(node, param_name);
}

void NodeSyncMgr::updateNodeVisibility(NodeLocation& node_location) {
    auto graph_model = zenoApp->graphsManager()->currentModel();
    if (!graph_model) {
        return;
    }

    const QModelIndex& nodeIdx = node_location.node;
    if (!nodeIdx.isValid())
        return;

    auto node_id = nodeIdx.data(ROLE_NODE_NAME).toString();
    int old_option = nodeIdx.data(ROLE_NODE_STATUS).toInt();
    int new_option = old_option;
    new_option ^= zeno::View;

    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(nodeIdx.model());
    pModel->setData(nodeIdx, new_option, ROLE_NODE_STATUS);
}

void NodeSyncMgr::updateNodeInputString(NodeLocation node_location,
                                        const std::string& input_name,
                                        const std::string& new_value) {
    auto graph_model = zenoApp->graphsManager()->currentModel();
    if (!graph_model) {
        return;
    }

    const QModelIndex& nodeIdx = node_location.node;
    if (!nodeIdx.isValid())
        return;

    auto node_id = nodeIdx.data(ROLE_NODE_NAME).toString();
    auto inputs = nodeIdx.data(ROLE_INPUTS).value<PARAMS_INFO>();
    const QString& paramName = QString::fromStdString(input_name);
    if (inputs.find(paramName) != inputs.end()) {
        auto& param = inputs[paramName];
        param.defl = new_value;
    }
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(nodeIdx.model());
    pModel->setData(nodeIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
}

void NodeSyncMgr::updateNodeParamString(NodeLocation node_location,
                                        const std::string& param_name,
                                        const std::string& new_value) {
    updateNodeInputString(node_location, param_name, new_value);
}

std::string NodeSyncMgr::getPrimSockName(const NodeLocation& node_location) {
    auto& nodename = node_location.node.data(ROLE_NODE_NAME).toString().toStdString();
    if (m_prim_sock_map.find(nodename) != m_prim_sock_map.end())
        return m_prim_sock_map[nodename];
    else {
        PARAMS_INFO outputs = node_location.node.data(ROLE_OUTPUTS).value<PARAMS_INFO>();
        outputs.remove("DST");
        if (!outputs.empty())
            return outputs.firstKey().toStdString();
        else
            return "prim";
    }
}

}