#include "command.h"
#include "variantptr.h"
#include "reflect/reflection.generated.hpp"


AddNodeCommand::AddNodeCommand(const QString& cate, zeno::NodeData& nodedata, QStringList& graphPath)
    : QUndoCommand()
    , m_model(GraphsManager::instance().getGraph(graphPath))
    , m_graphPath(graphPath)
    , m_nodeData(nodedata)
    , m_pos(nodedata.uipos)
    , m_cate(cate)
{
    if (m_nodeData.cls == "Subnet") //init subnet default socket
    {
        zeno::ParamTab tab;
        zeno::ParamGroup default;

        zeno::ParamUpdateInfo info;
        zeno::ParamPrimitive param;
        param.bInput = true;
        param.name = "int1";
        param.defl = zeno::reflect::make_any<int>(0);
        param.type = zeno::types::gParamType_Int;
        info.param = param;
        default.params.push_back(param);
        param.bInput = false;
        param.name = "output1";
        param.type = Param_Null;
        param.socketType = zeno::Socket_Primitve;
        info.param = param;
        zeno::ParamObject objInput;
        objInput.bInput = true;
        objInput.name = "objInput1";
        objInput.type = Param_Object;
        objInput.socketType = zeno::Socket_ReadOnly;
        zeno::ParamObject objOutput;
        objOutput.bInput = false;
        objOutput.name = "objOutput1";
        objOutput.type = Param_Object;
        objOutput.socketType = zeno::Socket_Output;

        tab.groups.emplace_back(std::move(default));
        m_nodeData.customUi.inputPrims.tabs.emplace_back(std::move(tab));
        m_nodeData.customUi.inputObjs.push_back(objInput);
        m_nodeData.customUi.outputPrims.push_back(param);
        m_nodeData.customUi.outputObjs.push_back(objOutput);
    }
}

AddNodeCommand::~AddNodeCommand()
{
}

void AddNodeCommand::redo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model) {
        m_nodeData.uipos = m_pos;
        m_nodeData = m_model->_createNodeImpl(m_cate, m_nodeData, false);
    }
}

void AddNodeCommand::undo()
{
    if (m_model) {
        auto nodename = QString::fromStdString(m_nodeData.name);
        if (auto spnode = m_model->getWpNode(nodename).lock())
        {
            m_pos = spnode->get_pos();
        }
        m_model->_removeNodeImpl(nodename);
    }
}


zeno::NodeData AddNodeCommand::getNodeData()
{
    return m_nodeData;
}

RemoveNodeCommand::RemoveNodeCommand(zeno::NodeData& nodeData, QStringList& graphPath)
    : QUndoCommand()
    , m_model(GraphsManager::instance().getGraph(graphPath))
    , m_nodeData(nodeData)
    , m_graphPath(graphPath)
    , m_cate("")
{
    //m_id = m_data[ROLE_OBJID].toString();

    ////all links will be removed when remove node, for caching other type data,
    ////we have to clean the data here.
    //OUTPUT_SOCKETS outputs = m_data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    //INPUT_SOCKETS inputs = m_data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    //for (auto it = outputs.begin(); it != outputs.end(); it++)
    //{
    //    it->second.info.links.clear();
    //}
    //for (auto it = inputs.begin(); it != inputs.end(); it++)
    //{
    //    it->second.info.links.clear();
    //}
    //m_data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    //m_data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
}

RemoveNodeCommand::~RemoveNodeCommand()
{
}

void RemoveNodeCommand::redo()
{
    if (m_model) {
        auto nodename = QString::fromStdString(m_nodeData.name);
        auto spNode = m_model->getWpNode(nodename).lock();
        if (std::shared_ptr<zeno::SubnetNode> subnetNode = std::dynamic_pointer_cast<zeno::SubnetNode>(spNode)) {   //if is subnet/assets£¬record cate
            m_cate = subnetNode->isAssetsNode() ? "assets" : "";
        }
        m_model->_removeNodeImpl(QString::fromStdString(m_nodeData.name));
    }
}

void RemoveNodeCommand::undo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model)
        m_nodeData = m_model->_createNodeImpl(m_cate, m_nodeData, false);
}

LinkCommand::LinkCommand(bool bAddLink, const zeno::EdgeInfo& link, QStringList& graphPath)
    : QUndoCommand()
    , m_bAdd(bAddLink)
    , m_link(link)
    , m_model(GraphsManager::instance().getGraph(graphPath))
    , m_graphPath(graphPath)
{
}

void LinkCommand::redo()
{
    if (m_bAdd)
    {
        m_model = GraphsManager::instance().getGraph(m_graphPath);
        if (m_model)
            m_model->_addLinkImpl(m_link);
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
        if (m_model)
            m_model->_removeLinkImpl(m_link);
    }
    else
    {
        m_model = GraphsManager::instance().getGraph(m_graphPath);
        if (m_model)
            m_model->_addLinkImpl(m_link);
    }
}

ModelDataCommand::ModelDataCommand(const QModelIndex& index, const QVariant& oldData, const QVariant& newData, int role, QStringList& graphPath)
    : m_oldData(oldData)
    , m_newData(newData)
    , m_role(role)
    , m_graphPath(graphPath)
    , m_model(nullptr)
    , m_nodeName("")
    , m_paramName("")
{
    m_nodeName = index.data(ROLE_NODE_NAME).toString();
    if (m_role == ROLE_PARAM_VALUE)  //index of paramsModel, need record paramName
        m_paramName = index.data(ROLE_PARAM_NAME).toString();
}

void ModelDataCommand::redo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model)
    {
        auto nodeIdx = m_model->indexFromName(m_nodeName);
        if (nodeIdx.isValid())
        {
            if (m_role == ROLE_OBJPOS || m_role == ROLE_COLLASPED)
            {
                m_model->setData(nodeIdx, m_newData, m_role);
            }else if (m_role == ROLE_PARAM_VALUE)
            {
                if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(nodeIdx.data(ROLE_PARAMS)))
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
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model)
    {
        auto nodeIdx = m_model->indexFromName(m_nodeName);
        if (nodeIdx.isValid())
        {
            if (m_role == ROLE_OBJPOS || m_role == ROLE_COLLASPED)
            {
                m_model->setData(nodeIdx, m_oldData, m_role);
            }
            else if (m_role == ROLE_PARAM_VALUE)
            {
                if (ParamsModel* paramsModel = QVariantPtr<ParamsModel>::asPtr(nodeIdx.data(ROLE_PARAMS)))
                {
                    auto paramIdx = paramsModel->paramIdx(m_paramName, true);
                    paramsModel->setData(paramIdx, m_oldData, m_role);
                }
            }
        }
    }
}

NodeStatusCommand::NodeStatusCommand(bool isSetView, const QString& name, bool bOn, QStringList& graphPath)
    : m_On(bOn)
    , m_graphPath(graphPath)
    , m_model(nullptr)
    , m_nodeName(name)
    , m_isSetView(isSetView)
{
}

void NodeStatusCommand::redo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model)
    {
        auto idx = m_model->indexFromName(m_nodeName);
        if (idx.isValid()) {
            if (m_isSetView)
            {
                m_model->_setViewImpl(idx, m_On);
            }
            else {
                //m_model->_setMuteImpl(idx, m_On);
            }
        }
    }
}

void NodeStatusCommand::undo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model)
    {
        auto idx = m_model->indexFromName(m_nodeName);
        if (idx.isValid()) {
            if (m_isSetView)
            {
                m_model->_setViewImpl(idx, !m_On);
            }
            else {
                //m_model->_setMuteImpl(idx, !m_On);
            }
        }
    }
}
