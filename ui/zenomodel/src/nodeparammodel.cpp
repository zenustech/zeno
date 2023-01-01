#include "nodeparammodel.h"
#include "vparamitem.h"
#include "modelrole.h"


NodeParamModel::NodeParamModel(const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent)
    : ViewParamModel(true, nodeIdx, pModel, parent)
    , m_inputs(nullptr)
    , m_params(nullptr)
    , m_outputs(nullptr)
{
    initUI();
}

NodeParamModel::~NodeParamModel()
{
}

void NodeParamModel::initUI()
{
    /* structure:
      invisibleroot
            |-- Inputs (Group)
                -- input param1 (Item)
                -- input param2
                ...

            |-- Params (Group)
                -- param1 (Item)
                -- param2 (Item)

            |- Outputs (Group)
                - output param1 (Item)
                - output param2 (Item)
    */
    m_inputs = new VParamItem(VPARAM_GROUP, "inputs");
    m_params = new VParamItem(VPARAM_GROUP, "params");
    m_outputs = new VParamItem(VPARAM_GROUP, "outputs");
    appendRow(m_inputs);
    appendRow(m_params);
    appendRow(m_outputs);
}

bool NodeParamModel::getInputSockets(INPUT_SOCKETS& inputs)
{
    for (int r = 0; r < m_inputs->rowCount(); r++)
    {
        VParamItem* param = static_cast<VParamItem*>(m_inputs->child(r));
        const QString& name = param->m_name;

        INPUT_SOCKET inSocket;
        inSocket.info.defaultValue = param->m_value;
        inSocket.info.nodeid = m_nodeIdx.data(ROLE_OBJID).toString();
        inSocket.info.name = param->m_name;
        inSocket.info.type = param->m_type;
        inSocket.info.sockProp = param->m_sockProp;
        inSocket.info.links = exportLinks(param->m_links);

        //todo: dict key model as children of this item?
        inputs.insert(name, inSocket);
    }
    return true;
}

bool NodeParamModel::getOutputSockets(OUTPUT_SOCKETS& outputs)
{
    for (int r = 0; r < m_outputs->rowCount(); r++)
    {
        VParamItem* param = static_cast<VParamItem*>(m_outputs->child(r));
        const QString& name = param->m_name;

        OUTPUT_SOCKET outSocket;
        outSocket.info.defaultValue = param->m_value;
        outSocket.info.nodeid = m_nodeIdx.data(ROLE_OBJID).toString();
        outSocket.info.name = name;
        outSocket.info.type = param->m_type;
        outSocket.info.sockProp = param->m_sockProp;
        outSocket.info.links = exportLinks(param->m_links);

        outputs.insert(name, outSocket);
    }
    return true;
}

bool NodeParamModel::getParams(PARAMS_INFO &params)
{
    for (int r = 0; r < m_params->rowCount(); r++)
    {
        VParamItem* param = static_cast<VParamItem*>(m_params->child(r));
        const QString& name = param->m_name;

        PARAM_INFO paramInfo;
        paramInfo.bEnableConnect = false;
        paramInfo.value = param->m_value;
        paramInfo.typeDesc = param->m_type;
        paramInfo.name = name;
        params.insert(name, paramInfo);
    }
    return true;
}

VParamItem* NodeParamModel::getInputs() const
{
    return m_inputs;
}

VParamItem* NodeParamModel::getParams() const
{
    return m_params;
}

VParamItem* NodeParamModel::getOutputs() const
{
    return m_outputs;
}

void NodeParamModel::setInputSockets(const INPUT_SOCKETS& inputs)
{
    for (INPUT_SOCKET inSocket : inputs)
    {
        VParamItem* pItem = new VParamItem(VPARAM_PARAM, inSocket.info.name, false);
        pItem->m_name = inSocket.info.name;
        pItem->m_value = inSocket.info.defaultValue;
        pItem->m_type = inSocket.info.type;
        pItem->m_sockProp = (SOCKET_PROPERTY)inSocket.info.sockProp;
        appendRow(pItem);
    }
}

void NodeParamModel::setParams(const PARAMS_INFO& params)
{
    for (PARAM_INFO paramInfo : params)
    {
        VParamItem* pItem = new VParamItem(VPARAM_PARAM, paramInfo.name, false);
        pItem->m_name = paramInfo.name;
        pItem->m_value = paramInfo.value;
        pItem->m_type = paramInfo.typeDesc;
        pItem->m_sockProp = SOCKPROP_UNKNOWN;
        appendRow(pItem);
    }
}

void NodeParamModel::setOutputSockets(const OUTPUT_SOCKETS& outputs)
{
    for (OUTPUT_SOCKET outSocket : outputs)
    {
        VParamItem *pItem = new VParamItem(VPARAM_PARAM, outSocket.info.name, false);
        pItem->m_name = outSocket.info.name;
        pItem->m_value = outSocket.info.defaultValue;
        pItem->m_type = outSocket.info.type;
        pItem->m_sockProp = (SOCKET_PROPERTY)outSocket.info.sockProp;
        appendRow(pItem);
    }
}

QList<EdgeInfo> NodeParamModel::exportLinks(const PARAM_LINKS& links)
{
    QList<EdgeInfo> linkInfos;
    for (auto linkIdx : links)
    {
        EdgeInfo link = exportLink(linkIdx);
        linkInfos.append(link);
    }
    return linkInfos;
}

EdgeInfo NodeParamModel::exportLink(const QModelIndex& linkIdx)
{
    EdgeInfo link;

    QModelIndex outSock = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
    QModelIndex inSock = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    ZASSERT_EXIT(outSock.isValid() && inSock.isValid(), link);

    QModelIndex outCoreParam = outSock.data(ROLE_PARAM_COREIDX).toModelIndex();
    QModelIndex inCoreParam = inSock.data(ROLE_PARAM_COREIDX).toModelIndex();
    ZASSERT_EXIT(outCoreParam.isValid() && inCoreParam.isValid(), link);

    //for dict panel socket, write the full path of output socket.
    if (outSock.data(ROLE_PARAM_CLASS) == PARAM_INNER_OUTPUT) {
        link.outSockPath = outSock.data(ROLE_OBJPATH).toString();
    } else {
        link.outSockPath = outCoreParam.data(ROLE_OBJPATH).toString();
    }

    if (inSock.data(ROLE_PARAM_CLASS) == PARAM_INNER_INPUT) {
        link.inSockPath = inSock.data(ROLE_OBJPATH).toString();
    } else {
        link.inSockPath = inCoreParam.data(ROLE_OBJPATH).toString();
    }
    return link;
}

void NodeParamModel::setParam(
                PARAM_CLASS cls,
                const QString& name,
                const QString& type,
                const QVariant& deflValue,
                SOCKET_PROPERTY prop)
{
    VParamItem *pItem = nullptr;
    if (PARAM_INPUT == cls)
    {
        if (!(pItem = m_inputs->getItem(name)))
        {
            pItem = new VParamItem(VPARAM_PARAM, name);
            m_inputs->appendRow(pItem);
        }
    }
    else if (PARAM_PARAM == cls)
    {
        if (!(pItem = m_params->getItem(name)))
        {
            pItem = new VParamItem(VPARAM_PARAM, name);
            m_params->appendRow(pItem);
        }
    }
    else if (PARAM_OUTPUT == cls)
    {
        if (!(pItem = m_outputs->getItem(name)))
        {
            pItem = new VParamItem(VPARAM_PARAM, name);
            m_outputs->appendRow(pItem);
        }
    }

    pItem->m_name = name;
    pItem->m_value = deflValue;
    pItem->m_sockProp = prop;
}

QVariant NodeParamModel::getValue(PARAM_CLASS cls, const QString& name) const
{
    VParamItem *pItem = nullptr;
    if (PARAM_INPUT == cls)
    {
        if (!(pItem = m_inputs->getItem(name)))
        {
            return QVariant();
        }
    }
    else if (PARAM_PARAM == cls)
    {
        if (!(pItem = m_params->getItem(name)))
        {
            return QVariant();
        }
    }
    else if (PARAM_OUTPUT == cls)
    {
        if (!(pItem = m_outputs->getItem(name)))
        {
            return QVariant();
        }
    }
    else
    {
        return QVariant();
    }
    return pItem->m_value;
}


QVariant NodeParamModel::data(const QModelIndex& index, int role) const
{
    switch (role)
    {
    case ROLE_INPUT_MODEL:
    case ROLE_PARAM_MODEL:
    case ROLE_OUTPUT_MODEL:
        //legacy interface.
        return QVariant();
    default:
        return ViewParamModel::data(index, role);
    }

}

bool NodeParamModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return ViewParamModel::setData(index, value, role);
}


