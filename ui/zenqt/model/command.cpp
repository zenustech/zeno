#include "command.h"
#include "variantptr.h"


AddNodeCommand::AddNodeCommand(zeno::NodeData& nodedata, QStringList& graphPath)
    : QUndoCommand()
    , m_model(GraphsManager::instance().getGraph(graphPath))
    , m_graphPath(graphPath)
    , m_nodeData(nodedata)
{
    if (m_nodeData.cls == "Subnet") //init subnet default socket
    {
        zeno::ParamTab tab;
        zeno::ParamGroup default;

        zeno::ParamsUpdateInfo updateInfo;
        zeno::ParamUpdateInfo info;
        info.param.bInput = true;
        info.param.name = "input1";
        info.param.socketType = zeno::PrimarySocket;
        updateInfo.push_back(info);
        info.param.bInput = true;
        info.param.name = "input2";
        info.param.socketType = zeno::PrimarySocket;
        updateInfo.push_back(info);
        info.param.bInput = false;
        info.param.name = "output1";
        info.param.socketType = zeno::PrimarySocket;
        updateInfo.push_back(info);

        default.params.push_back(updateInfo[0].param);
        default.params.push_back(updateInfo[1].param);
        tab.groups.emplace_back(std::move(default));
        m_nodeData.customUi.tabs.emplace_back(std::move(tab));
        m_nodeData.customUi.outputs.push_back(updateInfo[2].param);
    }
}

AddNodeCommand::~AddNodeCommand()
{
}

void AddNodeCommand::redo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model) {
        m_nodeData = m_model->_createNodeImpl(m_nodeData, m_model->m_wpCoreGraph, zeno::GraphData(), false);
    }
}

void AddNodeCommand::undo()
{
    if (m_model) {
        auto nodename = QString::fromStdString(m_nodeData.name);
        m_model->_removeNodeImpl(nodename, m_model->m_wpCoreGraph);
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
    , m_graphData()
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
        if (m_nodeData.cls == "Subnet")  //如果删除subnetnode，记录graphData，undo时恢复
        {
            auto& it = m_model->m_name2uuid.find(nodename);
            if (it != m_model->m_name2uuid.end() && m_model->m_nodes.find(it.value()) != m_model->m_nodes.end())
            {
                auto wpSubnetNode = m_model->m_nodes[it.value()]->m_wpNode;
                if (auto spSubnetNode = wpSubnetNode.lock())
                {
                    if (std::shared_ptr<zeno::SubnetNode> subnetNode = std::dynamic_pointer_cast<zeno::SubnetNode>(spSubnetNode)) {
                        m_graphData = subnetNode->subgraph->exportGraph();
                    }
                }
            }
        }
        m_model->_removeNodeImpl(QString::fromStdString(m_nodeData.name), m_model->m_wpCoreGraph);
    }
}

void RemoveNodeCommand::undo()
{
    m_model = GraphsManager::instance().getGraph(m_graphPath);
    if (m_model)
        m_nodeData = m_model->_createNodeImpl(m_nodeData, m_model->m_wpCoreGraph, m_graphData, false);
}