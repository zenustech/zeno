#include <QtWidgets>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/igraphsmodel.h>
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

void TransferAcceptor::setFilePath(const QString& fileName)
{

}

void TransferAcceptor::switchSubGraph(const QString& graphName)
{

}

bool TransferAcceptor::addNode(const QString& nodeid, const QString& name, const NODE_DESCS& descriptors)
{
    if (m_nodes.find(nodeid) != m_nodes.end())
        return false;

    NODE_DATA data;
    data[ROLE_OBJID] = nodeid;
    data[ROLE_OBJNAME] = name;
    data[ROLE_COLLASPED] = false;
    data[ROLE_NODETYPE] = NodesMgr::nodeType(name);

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
    const QString& nodeName = data[ROLE_OBJNAME].toString();
    if (nodeName == "MakeDict")
    {
        INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        for (auto keyName : keys) {
            addDictKey(id, keyName, true);
        }
    } else if (nodeName == "ExtractDict") {
        OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (auto keyName : keys) {
            addDictKey(id, keyName, false);
        }
    }
}

void TransferAcceptor::initSockets(const QString& id, const QString& name, const NODE_DESCS& legacyDescs)
{
    NODE_DESC desc;
    bool ret = m_pModel->getDescriptor(name, desc);
    ZASSERT_EXIT(ret);
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());

    //params
    INPUT_SOCKETS inputs;
    PARAMS_INFO params;
    OUTPUT_SOCKETS outputs;

    for (PARAM_INFO descParam : desc.params)
    {
        PARAM_INFO param;
        param.name = descParam.name;
        param.control = descParam.control;
        param.typeDesc = descParam.typeDesc;
        param.defaultValue = descParam.defaultValue;
        params.insert(param.name, param);
    }
    for (INPUT_SOCKET descInput : desc.inputs)
    {
        INPUT_SOCKET input;
        input.info.nodeid = id;
        input.info.control = descInput.info.control;
        input.info.type = descInput.info.type;
        input.info.name = descInput.info.name;
        input.info.defaultValue = descInput.info.defaultValue;
        inputs.insert(input.info.name, input);
    }
    for (OUTPUT_SOCKET descOutput : desc.outputs)
    {
        OUTPUT_SOCKET output;
        output.info.nodeid = id;
        output.info.control = descOutput.info.control;
        output.info.type = descOutput.info.type;
        output.info.name = descOutput.info.name;
        outputs[output.info.name] = output;
    }

    NODE_DATA& data = m_nodes[id];
    data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    data[ROLE_PARAMETERS] = QVariant::fromValue(params);
    data[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(NodesMgr::initParamsNotDesc(name));
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
        INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        if (inputs.find(keyName) == inputs.end())
        {
            INPUT_SOCKET inputSocket;
            inputSocket.info.name = keyName;
            inputSocket.info.nodeid = id;
            inputSocket.info.control = CONTROL_NONE;
            inputSocket.info.sockProp = SOCKPROP_EDITABLE;
            inputSocket.info.type = "";
            inputs[keyName] = inputSocket;
            data[ROLE_INPUTS] = QVariant::fromValue(inputs);
        }
    }
    else
    {
        OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        if (outputs.find(keyName) == outputs.end())
        {
            OUTPUT_SOCKET outputSocket;
            outputSocket.info.name = keyName;
            outputSocket.info.nodeid = id;
            outputSocket.info.control = CONTROL_NONE;
            outputSocket.info.sockProp = SOCKPROP_EDITABLE;
            outputSocket.info.type = "";
            outputs[keyName] = outputSocket;
            data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
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
    INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    if (inputs.find(sockName) != inputs.end())
    {
        inputs[sockName].info.dictpanel.bCollasped = bCollasped;
        data[ROLE_INPUTS] = QVariant::fromValue(inputs);
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
    INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    if (inputs.find(sockName) != inputs.end())
    {
        INPUT_SOCKET& inSocket = inputs[sockName];
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
        data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    }

    OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    if (outputs.find(sockName) != outputs.end())
    {
        OUTPUT_SOCKET& outSocket = outputs[sockName];
        DICTKEY_INFO item;
        item.key = keyName;

        QString newKeyPath = "[node]/outputs/" + sockName + "/" + keyName;
        outSocket.info.dictpanel.keys.append(item);
        //no need to import link here.
        data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
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
    bool ret = m_pModel->getDescriptor(nodeCls, desc);
    ZASSERT_EXIT(ret);

    //parse default value.
    QVariant defaultValue;
    if (!defaultVal.IsNull())
    {
        SOCKET_INFO descInfo;
        if (desc.inputs.find(inSock) != desc.inputs.end()) {
            descInfo = desc.inputs[inSock].info;
        }

        defaultValue = UiHelper::parseJsonByType(descInfo.type, defaultVal, nullptr);
    }

    ZASSERT_EXIT(m_nodes.find(inNode) != m_nodes.end());
    NODE_DATA& data = m_nodes[inNode];

    //standard inputs desc by latest descriptors. 
    INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    if (inputs.find(inSock) != inputs.end())
    {
        if (!defaultValue.isNull())
            inputs[inSock].info.defaultValue = defaultValue;
        if (!outLinkPath.isEmpty())
        {
            const QString& inSockPath = UiHelper::constructObjPath(m_currSubgraph, inNode, "[node]/inputs/", inSock);
            EdgeInfo info(outLinkPath, inSockPath);
            inputs[inSock].info.links.append(info);
            m_links.append(info);
        }
        data[ROLE_INPUTS] = QVariant::fromValue(inputs);
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
            inputs[inSock] = inSocket;

            if (!outLinkPath.isEmpty())
            {
                const QString& inSockPath = UiHelper::constructObjPath(m_currSubgraph, inNode, "[node]/inputs/", inSock);
                EdgeInfo info(outLinkPath, inSockPath);
                inputs[inSock].info.links.append(info);
                m_links.append(info);
            }
            data[ROLE_INPUTS] = QVariant::fromValue(inputs);
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
    INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    if (inputs.find(inSock) != inputs.end()) {
        inputs[inSock].info.control = control;
        inputs[inSock].info.ctrlProps = ctrlProperties.toMap();
        data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    }
}

void TransferAcceptor::setToolTip(PARAM_CLASS cls, const QString & inNode, const QString & inSock, const QString & toolTip)
{
    ZASSERT_EXIT(m_nodes.find(inNode) != m_nodes.end());
    NODE_DATA &data = m_nodes[inNode];
    
    if (cls == PARAM_INPUT) 
    {
        INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        if (inputs.find(inSock) != inputs.end()) 
        {
            inputs[inSock].info.toolTip = toolTip;
            data[ROLE_INPUTS] = QVariant::fromValue(inputs);
        }
    } 
    else if (cls == PARAM_OUTPUT) 
    {
        OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        if (outputs.find(inSock) != outputs.end()) 
        {
            outputs[inSock].info.toolTip = toolTip;
            data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
        }
    } 
}
void TransferAcceptor::setParamValue(const QString &id, const QString &nodeCls, const QString &name,const rapidjson::Value &value) {
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA& data = m_nodes[id];

    NODE_DESC desc;
    bool ret = m_pModel->getDescriptor(nodeCls, desc);
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
            var = UiHelper::parseJsonByValue(paramInfo.typeDesc, value, nullptr);
        else
            var = UiHelper::parseJsonByType(paramInfo.typeDesc, value, nullptr);
    }

    PARAMS_INFO params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
    if (params.find(name) != params.end())
    {
        zeno::log_trace("found param name {}", name.toStdString());
        params[name].value = var;
        data[ROLE_PARAMETERS] = QVariant::fromValue(params);
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
            params[name] = paramData;
            data[ROLE_PARAMETERS] = QVariant::fromValue(params);
            return;
        }
        if (nodeCls == "MakeHeatmap" && name == "_RAMPS")
        {
            PARAM_INFO paramData;
            paramData.control = CONTROL_COLOR;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            params[name] = paramData;
            data[ROLE_PARAMETERS] = QVariant::fromValue(params);
            return;
        }

        PARAMS_INFO _params = data[ROLE_PARAMS_NO_DESC].value<PARAMS_INFO>();
        _params[name].value = var;
        data[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(_params);
        zeno::log_warn("not found param name {}", name.toStdString());
    }
}

void TransferAcceptor::setParamValue2(const QString &id, const QString &noCls,const PARAMS_INFO &params) 
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA &data = m_nodes[id];

	if (!params.isEmpty()) 
	{
        data[ROLE_PARAMETERS] = QVariant::fromValue(params);
    }
}

void TransferAcceptor::setPos(const QString& id, const QPointF& pos)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id][ROLE_OBJPOS] = pos;
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
            data[ROLE_COLLASPED] = true;
        }
    }
    data[ROLE_OPTIONS] = opts;
}

void TransferAcceptor::setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps)
{

}

void TransferAcceptor::setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    NODE_DATA &data = m_nodes[id];
    PARAMS_INFO paramsNotDesc;
    paramsNotDesc["blackboard"].name = "blackboard";
    paramsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
    data[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(paramsNotDesc);
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
        PARAMS_INFO params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
        ZASSERT_EXIT(params.find("name") != params.end() &&
            params.find("type") != params.end() &&
            params.find("defl") != params.end());

        const QString& descType = params["type"].value.toString();
        PARAM_INFO& defl = params["defl"];
        defl.control = UiHelper::getControlByType(descType);
        defl.value = UiHelper::parseVarByType(descType, defl.value, nullptr);
        defl.typeDesc = descType;
        data[ROLE_PARAMETERS] = QVariant::fromValue(params);
    }
}

void TransferAcceptor::addCustomUI(const QString& id, const VPARAM_INFO& invisibleRoot)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id][ROLE_CUSTOMUI_PANEL_IO] = QVariant::fromValue(invisibleRoot);
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
