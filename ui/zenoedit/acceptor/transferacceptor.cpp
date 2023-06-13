#include <QtWidgets>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/graphsmanagment.h>
#include "transferacceptor.h"
#include <zeno/utils/logger.h>
#include "util/log.h"
#include <zenoio/reader/zsgreader.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/uihelper.h>


TransferAcceptor::TransferAcceptor(IGraphsModel* pModel)
    : m_pModel(pModel)
{

}

bool TransferAcceptor::setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS& legacyDescs)
{
    return false;
}

void TransferAcceptor::BeginSubgraph(const QString &name)
{
    //no cache, for data consistency.
    m_currSubgraph = name;
    m_links.clear();
    m_nodes.clear();
}

void TransferAcceptor::EndSubgraph()
{
    m_currSubgraph = "";
}

void TransferAcceptor::EndGraphs()
{
}

bool TransferAcceptor::setCurrentSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx)
{
    m_currSubgraph = subgIdx.data(ROLE_OBJNAME).toString();
    return true;
}

void TransferAcceptor::switchSubGraph(const QString& graphName)
{

}

bool TransferAcceptor::addNode(const QString& nodeid, const QString& name, const QString& customName, const NODE_DESCS& descriptors)
{
    if (m_nodes.find(nodeid) != m_nodes.end())
        return false;

    NODE_DATA data;
    data.ident = nodeid;
    data.nodeCls = name;
    data.customName = customName;
    data.bCollasped = false;
    data.type = GraphsManagment::instance().nodeType(name);

    m_nodes.insert(nodeid, data);
    return true;
}

void TransferAcceptor::setViewRect(const QRectF& rc)
{

}

void TransferAcceptor::setSocketKeys(const QString& id, const QStringList& keys)
{
    ZASSERT_EXIT(m_nodes.find(id) == m_nodes.end());
    NODE_DATA& data = m_nodes[id];
    const QString& nodeName = data.nodeCls;
    if (nodeName == "MakeDict")
    {
        INPUT_SOCKETS inputs = data.inputs;
        for (auto keyName : keys) {
            addDictKey(id, keyName, true);
        }
    } else if (nodeName == "ExtractDict") {
        OUTPUT_SOCKETS outputs = data.outputs;
        for (auto keyName : keys) {
            addDictKey(id, keyName, false);
        }
    }
}

void TransferAcceptor::initSockets(const QString& id, const QString& name, const NODE_DESCS& legacyDescs)
{
    NODE_DESC desc;
    auto &mgr = GraphsManagment::instance();
    bool ret = mgr.getDescriptor(name, desc);
    ZASSERT_EXIT(ret);
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());

    NODE_DATA &data = m_nodes[id];

    for (PARAM_INFO descParam : desc.params)
    {
        PARAM_INFO param;
        param.name = descParam.name;
        param.control = descParam.control;
        param.typeDesc = descParam.typeDesc;
        param.defaultValue = descParam.defaultValue;
        data.params.insert(param.name, param);
    }
    for (INPUT_SOCKET descInput : desc.inputs)
    {
        INPUT_SOCKET input;
        input.info.nodeid = id;
        input.info.control = descInput.info.control;
        input.info.type = descInput.info.type;
        input.info.name = descInput.info.name;
        input.info.defaultValue = descInput.info.defaultValue;
        data.inputs.insert(input.info.name, input);
    }
    for (OUTPUT_SOCKET descOutput : desc.outputs)
    {
        OUTPUT_SOCKET output;
        output.info.nodeid = id;
        output.info.control = descOutput.info.control;
        output.info.type = descOutput.info.type;
        output.info.name = descOutput.info.name;
        data.outputs[output.info.name] = output;
    }
    data.parmsNotDesc = NodesMgr::initParamsNotDesc(name);
}

void TransferAcceptor::addSocket(bool bInput, const QString& ident, const QString& sockName, const QString& sockProperty)
{

}

void TransferAcceptor::addDictKey(const QString& id, const QString& keyName, bool bInput)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());

    NODE_DATA &data = m_nodes[id];
    if (bInput)
    {
        if (data.inputs.find(keyName) == data.inputs.end())
        {
            INPUT_SOCKET inputSocket;
            inputSocket.info.name = keyName;
            inputSocket.info.nodeid = id;
            inputSocket.info.control = CONTROL_NONE;
            inputSocket.info.sockProp = SOCKPROP_EDITABLE;
            inputSocket.info.type = "";
            data.inputs[keyName] = inputSocket;
        }
    }
    else
    {
        if (data.outputs.find(keyName) == data.outputs.end())
        {
            OUTPUT_SOCKET outputSocket;
            outputSocket.info.name = keyName;
            outputSocket.info.nodeid = id;
            outputSocket.info.control = CONTROL_NONE;
            outputSocket.info.sockProp = SOCKPROP_EDITABLE;
            outputSocket.info.type = "";
            data.outputs[keyName] = outputSocket;
        }
    }
}

void TransferAcceptor::setInputSocket(
        const QString& nodeCls,
        const QString& inNode,
        const QString& inSock,
        const QString& outNode,
        const QString& outSock,
        const rapidjson::Value& defaultValue
)
{

}

void TransferAcceptor::setDictPanelProperty(
        bool bInput,
        const QString& ident,
        const QString& sockName,
        bool bCollasped
    )
{
    ZASSERT_EXIT(m_nodes.find(ident) != m_nodes.end());
    NODE_DATA& data = m_nodes[ident];

    //standard inputs desc by latest descriptors.
    if (data.inputs.find(sockName) != data.inputs.end())
    {
        data.inputs[sockName].info.dictpanel.bCollasped = bCollasped;
    }
}

void TransferAcceptor::addInnerDictKey(
        bool bInput,
        const QString& inNode,
        const QString& sockName,
        const QString& keyName,
        const QString& link
    )
{
    ZASSERT_EXIT(m_nodes.find(inNode) != m_nodes.end());
    NODE_DATA& data = m_nodes[inNode];

    //standard inputs desc by latest descriptors.
    if (data.inputs.find(sockName) != data.inputs.end())
    {
        INPUT_SOCKET& inSocket = data.inputs[sockName];
        DICTKEY_INFO item;
        item.key = keyName;

        QString newKeyPath = "[node]/inputs/" + sockName + "/" + keyName;
        QString inSockPath = UiHelper::constructObjPath(m_currSubgraph, inNode, newKeyPath);
        QString outSockPath = link;
        EdgeInfo edge(outSockPath, inSockPath);
        if (edge.isValid())
        {
            item.links.append(edge);
            m_links.append(edge);
        }
        inSocket.info.dictpanel.keys.append(item);
    }

    if (data.outputs.find(sockName) != data.outputs.end())
    {
        OUTPUT_SOCKET& outSocket = data.outputs[sockName];
        DICTKEY_INFO item;
        item.key = keyName;

        QString newKeyPath = "[node]/outputs/" + sockName + "/" + keyName;
        outSocket.info.dictpanel.keys.append(item);
        //no need to import link here.
    }
}

void TransferAcceptor::setInputSocket2(
                const QString& nodeCls,
                const QString& inNode,
                const QString& inSock,
                const QString& outLinkPath,
                const QString& sockProperty,
                const rapidjson::Value& defaultVal)
{
    NODE_DESC desc;
    auto &mgr = GraphsManagment::instance();
    bool ret = mgr.getDescriptor(nodeCls, desc);
    ZASSERT_EXIT(ret);

    //parse default value.
    QVariant defaultValue;
    if (!defaultVal.IsNull())
    {
        SOCKET_INFO descInfo;
        if (desc.inputs.find(inSock) != desc.inputs.end()) {
            descInfo = desc.inputs[inSock].info;
        }

        defaultValue = UiHelper::parseJsonByType(descInfo.type, defaultVal);
    }

    ZASSERT_EXIT(m_nodes.find(inNode) != m_nodes.end());
    NODE_DATA& data = m_nodes[inNode];

    //standard inputs desc by latest descriptors.
    if (data.inputs.find(inSock) != data.inputs.end())
    {
        if (!defaultValue.isNull())
            data.inputs[inSock].info.defaultValue = defaultValue;
        if (!outLinkPath.isEmpty())
        {
            const QString& inSockPath = UiHelper::constructObjPath(m_currSubgraph, inNode, "[node]/inputs/", inSock);
            EdgeInfo info(outLinkPath, inSockPath);
            data.inputs[inSock].info.links.append(info);
            m_links.append(info);
        }
    }
    else
    {
        //TODO: optimize the code.
        if (nodeCls == "MakeList" || nodeCls == "MakeDict")
        {
            INPUT_SOCKET inSocket;
            inSocket.info.name = inSock;
            if (nodeCls == "MakeDict")
            {
                inSocket.info.control = CONTROL_NONE;
                inSocket.info.sockProp = SOCKPROP_EDITABLE;
            }
            data.inputs[inSock] = inSocket;

            if (!outLinkPath.isEmpty())
            {
                const QString& inSockPath = UiHelper::constructObjPath(m_currSubgraph, inNode, "[node]/inputs/", inSock);
                EdgeInfo info(outLinkPath, inSockPath);
                data.inputs[inSock].info.links.append(info);
                m_links.append(info);
            }
        }
        else
        {
            zeno::log_warn("{}: no such input socket {}", nodeCls.toStdString(), inSock.toStdString());
        }
    }
}

void TransferAcceptor::setControlAndProperties(const QString& nodeCls, const QString& inNode, const QString& inSock, PARAM_CONTROL control, const QVariant& ctrlProperties)
{
    ZASSERT_EXIT(m_nodes.find(inNode) != m_nodes.end());
    NODE_DATA &data = m_nodes[inNode];

    //standard inputs desc by latest descriptors.
    if (data.inputs.find(inSock) != data.inputs.end()) {
        data.inputs[inSock].info.control = control;
        data.inputs[inSock].info.ctrlProps = ctrlProperties.toMap();
    }
}

void TransferAcceptor::setToolTip(PARAM_CLASS cls, const QString & inNode, const QString & inSock, const QString & toolTip)
{
    ZASSERT_EXIT(m_nodes.find(inNode) != m_nodes.end());
    NODE_DATA &data = m_nodes[inNode];
    
    if (cls == PARAM_INPUT) 
    {
        if (data.inputs.find(inSock) != data.inputs.end()) 
        {
            data.inputs[inSock].info.toolTip = toolTip;
        }
    } 
    else if (cls == PARAM_OUTPUT) 
    {
        if (data.outputs.find(inSock) != data.outputs.end()) 
        {
            data.outputs[inSock].info.toolTip = toolTip;
        }
    } 
}

void TransferAcceptor::setParamValue(const QString &id, const QString &nodeCls, const QString &name,const rapidjson::Value &value) {
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA& data = m_nodes[id];

    NODE_DESC desc;
    auto &mgr = GraphsManagment::instance();
    bool ret = mgr.getDescriptor(nodeCls, desc);
    ZASSERT_EXIT(ret);

    QVariant var;
    if (!value.IsNull())
    {
        PARAM_INFO paramInfo;
        if (desc.params.find(name) != desc.params.end()) {
            paramInfo = desc.params[name];
        }
        //todo: parentRef;
        if (nodeCls == "SubInput" || nodeCls == "SubOutput")
            var = UiHelper::parseJsonByValue(paramInfo.typeDesc, value);
        else
            var = UiHelper::parseJsonByType(paramInfo.typeDesc, value);
    }

    if (data.params.find(name) != data.params.end())
    {
        zeno::log_trace("found param name {}", name.toStdString());
        data.params[name].value = var;
    }
    else
    {
        if (nodeCls == "MakeCurvemap" && (name == "_POINTS" || name == "_HANDLERS"))
        {
            PARAM_INFO paramData;
            paramData.control = CONTROL_NONVISIBLE;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            data.params[name] = paramData;
            return;
        }
        if (nodeCls == "MakeHeatmap" && name == "_RAMPS")
        {
            PARAM_INFO paramData;
            paramData.control = CONTROL_COLOR;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            data.params[name] = paramData;
            return;
        }

        data.parmsNotDesc[name].value = var;
        zeno::log_warn("not found param name {}", name.toStdString());
    }
}

void TransferAcceptor::setParamValue2(const QString &id, const QString &noCls,const PARAMS_INFO &params) 
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA &data = m_nodes[id];

	if (!params.isEmpty()) 
	{
        data.params = params;
    }
}

void TransferAcceptor::setPos(const QString& id, const QPointF& pos)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id].pos = pos;
}

void TransferAcceptor::setOptions(const QString& id, const QStringList& options)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA &data = m_nodes[id];
    int opts = 0;
    for (int i = 0; i < options.size(); i++)
    {
        const QString& optName = options[i];
        if (optName == "ONCE")
        {
            opts |= OPT_ONCE;
        }
        else if (optName == "PREP")
        {
            opts |= OPT_PREP;
        }
        else if (optName == "VIEW")
        {
            opts |= OPT_VIEW;
        }
        else if (optName == "MUTE")
        {
            opts |= OPT_MUTE;
        }
        else if (optName == "collapsed")
        {
            data.bCollasped = true;
        }
    }
    data.options = opts;
}

void TransferAcceptor::setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps)
{

}

void TransferAcceptor::setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA &data = m_nodes[id];
    data.parmsNotDesc["blackboard"].name = "blackboard";
    data.parmsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
}

void TransferAcceptor::setTimeInfo(const TIMELINE_INFO& info)
{
}

TIMELINE_INFO TransferAcceptor::timeInfo() const
{
    return TIMELINE_INFO();
}

void TransferAcceptor::setLegacyCurve(
    const QString& id,
    const QVector<QPointF>& pts,
    const QVector<QPair<QPointF, QPointF>>& hdls)
{
}

QObject* TransferAcceptor::currGraphObj()
{
    return nullptr;
}

void TransferAcceptor::endInputs(const QString& id, const QString& nodeCls)
{

}

void TransferAcceptor::endParams(const QString& id, const QString& nodeCls)
{
    if (nodeCls == "SubInput" || nodeCls == "SubOutput")
    {
        NODE_DATA& data = m_nodes[id];
        ZASSERT_EXIT(data.params.find("name") != data.params.end() &&
            data.params.find("type") != data.params.end() &&
            data.params.find("defl") != data.params.end());

        const QString& descType = data.params["type"].value.toString();
        PARAM_INFO& defl = data.params["defl"];
        defl.control = UiHelper::getControlByType(descType);
        defl.value = UiHelper::parseVarByType(descType, defl.value);
        defl.typeDesc = descType;
    }
}

void TransferAcceptor::addCustomUI(const QString& id, const VPARAM_INFO& invisibleRoot)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id].customPanel = invisibleRoot;
}

QMap<QString, NODE_DATA> TransferAcceptor::nodes() const
{
    return m_nodes;
}

QList<EdgeInfo> TransferAcceptor::links() const
{
    return m_links;
}

void TransferAcceptor::getDumpData(QMap<QString, NODE_DATA>& nodes, QList<EdgeInfo>& links)
{
    nodes = m_nodes;
    links = m_links;
}

void TransferAcceptor::setIOVersion(zenoio::ZSG_VERSION versio)
{
}

void TransferAcceptor::resolveAllLinks()
{
}
