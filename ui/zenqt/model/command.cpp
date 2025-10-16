#include "command.h"
#include "variantptr.h"
#include <zeno/utils/helper.h>
#include <zeno/core/typeinfo.h>
#include <zeno/extra/SubnetNode.h>
#include "model/parammodel.h"
#include "util/uihelper.h"
#include "zenoapplication.h"


AddNodeCommand::AddNodeCommand(const QString& cate, zeno::NodeData& nodedata, const QStringList& graphPath)
    : QUndoCommand()
    , m_model(zenoApp->graphsManager()->getGraph(graphPath))
    , m_graphPath(graphPath)
    , m_nodeData(nodedata)
    , m_pos(nodedata.uipos)
    , m_cate(cate)
{
}

AddNodeCommand::~AddNodeCommand()
{
}

void AddNodeCommand::redo()
{
    m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
    if (m_model) {
        m_nodeData.uipos = m_pos;
        m_nodeData = m_model->_createNodeImpl(m_cate, m_nodeData, false);
    }
}

void AddNodeCommand::undo()
{
    if (m_model) {
        auto nodename = QString::fromStdString(m_nodeData.name);
        QModelIndex idx = m_model->indexFromName(nodename);
        QPointF pos = idx.data(QtRole::ROLE_OBJPOS).toPointF();
        m_pos = { pos.x(), pos.y() };
        m_model->_removeNodeImpl(nodename);
    }
}


zeno::NodeData AddNodeCommand::getNodeData()
{
    return m_nodeData;
}

RemoveNodeCommand::RemoveNodeCommand(zeno::NodeData& nodeData, const QStringList& graphPath)
    : QUndoCommand()
    , m_model(zenoApp->graphsManager()->getGraph(graphPath))
    , m_nodeData(nodeData)
    , m_graphPath(graphPath)
    , m_cate("")
{
    //m_id = m_data[ROLE_OBJID].toString();

    ////all links will be removed when remove node, for caching other type data,
    ////we have to clean the data here.
    //OUTPUT_SOCKETS outputs = m_data[QtRole::ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    //INPUT_SOCKETS inputs = m_data[QtRole::ROLE_INPUTS].value<INPUT_SOCKETS>();
    //for (auto it = outputs.begin(); it != outputs.end(); it++)
    //{
    //    it->second.info.links.clear();
    //}
    //for (auto it = inputs.begin(); it != inputs.end(); it++)
    //{
    //    it->second.info.links.clear();
    //}
    //m_data[QtRole::ROLE_INPUTS] = QVariant::fromValue(inputs);
    //m_data[QtRole::ROLE_OUTPUTS] = QVariant::fromValue(outputs);
}

RemoveNodeCommand::~RemoveNodeCommand()
{
}

void RemoveNodeCommand::redo()
{
    if (m_model) {
        auto nodename = QString::fromStdString(m_nodeData.name);
        QModelIndex idx = m_model->indexFromName(nodename);
        zeno::NodeType type = (zeno::NodeType)idx.data(QtRole::ROLE_NODETYPE).toInt();
        if (type == zeno::Node_AssetInstance) {
            m_cate = "assets";
        }
        else {
            m_cate = "";
        }
        //if (auto subnetNode = dynamic_cast<zeno::SubnetNode*>(spNode)) {   //if is subnet/assets，record cate
        //    m_cate = subnetNode->isAssetsNode() ? "assets" : "";
        //}
        m_model->_removeNodeImpl(QString::fromStdString(m_nodeData.name));
    }
}

void RemoveNodeCommand::undo()
{
    m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
    if (m_model)
        m_nodeData = m_model->_createNodeImpl(m_cate, m_nodeData, false);
}

LinkCommand::LinkCommand(bool bAddLink, const zeno::EdgeInfo& link, const QStringList& graphPath)
    : QUndoCommand()
    , m_bAdd(bAddLink)
    , m_link(link)
    , m_model(zenoApp->graphsManager()->getGraph(graphPath))
    , m_graphPath(graphPath)
{
}

void LinkCommand::redo()
{
    if (m_bAdd)
    {
        m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
        if (m_model) {
            QStringList outNodeLinkedNodes = UiHelper::findAllLinkdNodes(m_model, QString::fromStdString(m_link.inNode), false, true);
            bool outNodeChainHasViewNode = false;
            QString outNodeViewNode;
            for (auto& node : outNodeLinkedNodes) {
                auto nodeidx = m_model->indexFromName(node);
                if (nodeidx.data(QtRole::ROLE_NODE_ISVIEW).toBool()) {
                    outNodeChainHasViewNode = true;
                    outNodeViewNode = nodeidx.data(QtRole::ROLE_NODE_NAME).toString();
                    break;
                }
            }
            if (outNodeChainHasViewNode)
            {
                QStringList intNodeLinkedNodes = UiHelper::findAllLinkdNodes(m_model, QString::fromStdString(m_link.outNode), true, false);
                for (auto& node : intNodeLinkedNodes) {
                    auto nodeidx = m_model->indexFromName(node);
                    if (nodeidx.data(QtRole::ROLE_NODE_ISVIEW).toBool()) {
                        m_lastViewNodeName = nodeidx.data(QtRole::ROLE_NODE_NAME).toString();
                        m_model->_setViewImpl(nodeidx, false);
                        //break;
                    }
                }
            }
            m_model->_addLink_apicall(m_link);
    }
    }
    else
    {
        if (m_model)
            m_model->_removeLinkImpl(m_link);
    }
}

void LinkCommand::undo()
{
    if (m_bAdd)
    {
        if (m_model) {
            if (!m_lastViewNodeName.isEmpty()) {
                auto lastViewNodeIdx = m_model->indexFromName(m_lastViewNodeName);
                m_model->_setViewImpl(lastViewNodeIdx, true);
            }
            m_model->_removeLinkImpl(m_link);
    }
    }
    else
    {
        m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
        if (m_model)
            m_model->_addLink_apicall(m_link);
    }
}

ModelDataCommand::ModelDataCommand(const QModelIndex& index, const QVariant& oldData, const QVariant& newData, int role, const QStringList& graphPath)
    : m_oldData(oldData)
    , m_newData(newData)
    , m_role(role)
    , m_graphPath(graphPath)
    , m_model(nullptr)
    , m_nodeName("")
    , m_paramName("")
{
    m_nodeName = index.data(QtRole::ROLE_NODE_NAME).toString();
    if (m_role == QtRole::ROLE_PARAM_VALUE)  //index of paramsModel, need record paramName
        m_paramName = index.data(QtRole::ROLE_PARAM_NAME).toString();
}

void ModelDataCommand::redo()
{
    m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
    if (m_model)
    {
        auto nodeIdx = m_model->indexFromName(m_nodeName);
        if (nodeIdx.isValid())
        {
            if (m_role == QtRole::ROLE_OBJPOS || m_role == QtRole::ROLE_COLLASPED)
            {
                m_model->setData(nodeIdx, m_newData, m_role);
            }else if (m_role == QtRole::ROLE_PARAM_VALUE)
            {
                if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(nodeIdx.data(QtRole::ROLE_PARAMS)))
                {
                    auto paramIdx = paramsModel->paramIdx(m_paramName, true);
                    paramsModel->setData(paramIdx, m_newData, m_role);
                }
            }
        }
    }
}

void ModelDataCommand::undo()
{
    m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
    if (m_model)
    {
        auto nodeIdx = m_model->indexFromName(m_nodeName);
        if (nodeIdx.isValid())
        {
            if (m_role == QtRole::ROLE_OBJPOS || m_role == QtRole::ROLE_COLLASPED)
            {
                m_model->setData(nodeIdx, m_oldData, m_role);
            }
            else if (m_role == QtRole::ROLE_PARAM_VALUE)
            {
                if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(nodeIdx.data(QtRole::ROLE_PARAMS)))
                {
                    auto paramIdx = paramsModel->paramIdx(m_paramName, true);
                    paramsModel->setData(paramIdx, m_oldData, m_role);
                }
            }
        }
    }
}

NodeStatusCommand::NodeStatusCommand(zeno::NodeStatus status, const QString& name, bool bOn, const QStringList& graphPath)
    : m_bOn(bOn)
    , m_graphPath(graphPath)
    , m_model(nullptr)
    , m_nodeName(name)
    , m_status(status)
{
}

void NodeStatusCommand::redo()
{
    m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
    if (m_model)
    {
        auto idx = m_model->indexFromName(m_nodeName);
        if (idx.isValid()) {
            switch (m_status)
            {
            case zeno::View: {
                if (m_bOn) {
                    QStringList linkedNodes = UiHelper::findAllLinkdNodes(m_model, m_nodeName, true, true);
                    for (auto& node : linkedNodes) {
                        auto nodeidx = m_model->indexFromName(node);
                        if (nodeidx.data(QtRole::ROLE_NODE_ISVIEW).toBool() && nodeidx.data(QtRole::ROLE_NODE_NAME).toString() != m_nodeName) {
                            m_lastViewNodeName = nodeidx.data(QtRole::ROLE_NODE_NAME).toString();
                            m_model->_setViewImpl(nodeidx, !m_bOn);
                            //break;
                        }
                    }
                }
                m_model->_setViewImpl(idx, m_bOn);
                break;
            }
            case zeno::ByPass: {
                m_model->_setByPassImpl(idx, m_bOn);
                break;
            }
            case zeno::ClearSbn: {
                m_model->_setClearSubnetImpl(idx, m_bOn);
                break;
            }
            case zeno::Nocache: {
                m_model->_setNoCacheImpl(idx, m_bOn);
                break;
            }
            }
        }
    }
}

void NodeStatusCommand::undo()
{
    m_model = zenoApp->graphsManager()->getGraph(m_graphPath);
    if (m_model)
    {
        auto idx = m_model->indexFromName(m_nodeName);
        if (idx.isValid()) {
            switch (m_status)
            {
            case zeno::View: {
                if (m_bOn && !m_lastViewNodeName.isEmpty()) {
                    auto lastViewNodeIdx = m_model->indexFromName(m_lastViewNodeName);
                    m_model->_setViewImpl(lastViewNodeIdx, m_bOn);
                }
                m_model->_setViewImpl(idx, !m_bOn);
                break;
            }
            case zeno::ByPass: {
                m_model->_setByPassImpl(idx, !m_bOn);
                break;
            }
            case zeno::ClearSbn: {
                m_model->_setClearSubnetImpl(idx, !m_bOn);
                break;
            }
            case zeno::Nocache: {
                m_model->_setNoCacheImpl(idx, !m_bOn);
                break;
            }
            }
        }
    }
}
