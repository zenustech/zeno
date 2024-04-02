#include <QObject>
#include <QtWidgets>
#include <rapidjson/document.h>

#include "modelacceptor.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include <zeno/utils/logger.h>
#include "magic_enum.hpp"
#include "zassert.h"
#include "uihelper.h"
#include "variantptr.h"
#include "dictkeymodel.h"
#include <zenomodel/include/vparamitem.h>


ModelAcceptor::ModelAcceptor(GraphsModel* pModel, bool bImport)
    : m_pModel(pModel)
    , m_currentGraph(nullptr)
    , m_bImport(bImport)
{
}

bool ModelAcceptor::setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS& legacyDescs)
{
    //discard legacy desc except subnet desc.
    QList<NODE_DESC> subnetDescs;
    if (m_bImport)
    {
        for (const NODE_DESC& desc : legacyDescs)
        {
            subnetDescs.append(desc);
        }
    }
    else
    {
        QStringList subgraphs;
        for (const auto& subgraph : graphObj.GetObject())
        {
            if (subgraph.name != "main") {
                subgraphs.append(QString::fromUtf8(subgraph.name.GetString()));
            }
        }
        for (QString name : subgraphs)
        {
            if (legacyDescs.find(name) == legacyDescs.end())
            {
                zeno::log_warn("subgraph {} isn't described by the file descs.", name.toStdString());
                continue;
            }
            subnetDescs.append(legacyDescs[name]);
        }
    }
    bool ret = m_pModel->appendSubnetDescsFromZsg(subnetDescs, m_bImport);
    return ret;
}

void ModelAcceptor::setTimeInfo(const TIMELINE_INFO& info)
{
    m_timeInfo.beginFrame = qMin(info.beginFrame, info.endFrame);
    m_timeInfo.endFrame = qMax(info.beginFrame, info.endFrame);
    m_timeInfo.currFrame = qMax(qMin(m_timeInfo.currFrame, m_timeInfo.endFrame),
        m_timeInfo.beginFrame);
    m_timeInfo.timelinefps = info.timelinefps;
}

void ModelAcceptor::setRecordInfo(const RECORD_SETTING& info)
{
    m_recordInfo.record_path = info.record_path;
    m_recordInfo.videoname = info.videoname;
    m_recordInfo.fps = info.fps;
    m_recordInfo.bitrate = info.bitrate;
    m_recordInfo.numMSAA = info.numMSAA;
    m_recordInfo.numOptix = info.numOptix;
    m_recordInfo.width = info.width;
    m_recordInfo.height = info.height;
    m_recordInfo.bExportVideo = info.bExportVideo;
    m_recordInfo.needDenoise = info.needDenoise;
    m_recordInfo.bAutoRemoveCache = info.bAutoRemoveCache;
    m_recordInfo.bAov = info.bAov;
    m_recordInfo.bExr = info.bExr;
}

void ModelAcceptor::setLayoutInfo(const LAYOUT_SETTING& info)
{
    m_layoutInfo = info;
}

TIMELINE_INFO ModelAcceptor::timeInfo() const
{
    return m_timeInfo;
}

RECORD_SETTING ModelAcceptor::recordInfo() const
{
    return m_recordInfo;
}

LAYOUT_SETTING ModelAcceptor::layoutInfo() const
{
    return m_layoutInfo;
}

USERDATA_SETTING ModelAcceptor::userdataInfo() const
{
    return m_userdatInfo;
}

void ModelAcceptor::setUserDataInfo(const USERDATA_SETTING& info)
{
    m_userdatInfo = info;
}

void ModelAcceptor::BeginSubgraph(const QString& name, int type, bool bForkLocked)
{
    if (m_bImport && name == "main")
    {
        m_currentGraph = nullptr;
        return;
    }

    if (m_bImport)
        zeno::log_info("Importing subgraph {}", name.toStdString());

    ZASSERT_EXIT(m_pModel && !m_currentGraph);
    SubGraphModel* pSubModel = m_pModel->subGraph(name);
    if (pSubModel)
    {
        if (m_bImport)
        {
            for (int i = 0; i < pSubModel->rowCount(); i++)
            {
                QString ident = pSubModel->index(i, 0).data(ROLE_OBJID).toString();
                if (m_oldToNewNodeIds.contains(ident))
                    m_oldToNewNodeIds.remove(ident);
            }
            pSubModel->clear();
            zeno::log_warn("override subgraph {}", name.toStdString());
        }
    }
    else
    {
        pSubModel = new SubGraphModel(m_pModel);
        pSubModel->setName(name);
        pSubModel->setType((SUBGRAPH_TYPE)type);
        pSubModel->setForkLock(bForkLocked);
        m_pModel->appendSubGraph(pSubModel);
    }
    m_currentGraph = pSubModel;
}

bool ModelAcceptor::setCurrentSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx)
{
    ZASSERT_EXIT(pModel, false);
    m_pModel = qobject_cast<GraphsModel*>(pModel);
    ZASSERT_EXIT(m_pModel, false);
    m_currentGraph = m_pModel->subGraph(subgIdx.row());
    ZASSERT_EXIT(m_currentGraph, false);
    return true;
}

void ModelAcceptor::EndGraphs()
{
    resolveAllLinks();
}

void ModelAcceptor::resolveAllLinks()
{
    //add links on this subgraph.
    for (EdgeInfo link : m_subgLinks)
    {
        for (const auto &newId : m_oldToNewNodeIds.keys())
        {
            QString oldId = m_oldToNewNodeIds[newId];
            if (link.outSockPath.contains(oldId))
            {
                QString subgName = UiHelper::getSockSubgraph(link.inSockPath);
                QString paramPath = UiHelper::getParamPath(link.outSockPath);
                QString outSockPath = UiHelper::constructObjPath(subgName, newId, paramPath);
                link.outSockPath = outSockPath;
                break;
            }
        }
        QModelIndex inSock, outSock, inNode, outNode;
        QString subgName, inNodeCls, outNodeCls, inSockName, outSockName, paramCls;

        if (!link.outSockPath.isEmpty())
        {
            outSock = m_pModel->indexFromPath(link.outSockPath);
            outSockName = link.outSockPath;
        }
        if (!link.inSockPath.isEmpty())
        {
            inSock = m_pModel->indexFromPath(link.inSockPath);
            inSockName = link.inSockPath;
        }

        if (!inSock.isValid())
        {
            zeno::log_warn("no such input socket {} in {}", inSockName.toStdString(), inNodeCls.toStdString());
            continue;
        }
        if (!outSock.isValid())
        {
            zeno::log_warn("no such output socket {} in {}", outSockName.toStdString(), outNodeCls.toStdString());
            continue;
        }

        QModelIndex nodeIdx = outSock.data(ROLE_NODE_IDX).toModelIndex();
        if (!nodeIdx.isValid())
        {
            zeno::log_warn("cannot pull node index from outSock");
            continue;
        }
        QModelIndex subgIdx = nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
        ZASSERT_EXIT(subgIdx.isValid());

        if (NO_VERSION_NODE == nodeIdx.data(ROLE_NODETYPE))
            m_pModel->addLegacyLink(subgIdx, outSock, inSock);
        else
            m_pModel->addLink(subgIdx, outSock, inSock);
    }

    for (EdgeInfo link : m_subgLegacyLinks)
    {
        QModelIndex inSock, outSock, inNode, outNode;
        QString inNodeCls, outNodeCls, inSockName, outSockName, paramCls;

        if (!link.outSockPath.isEmpty())
        {
            outSock = m_pModel->indexFromPath(link.outSockPath);
        }
        if (!link.inSockPath.isEmpty())
        {
            inSock = m_pModel->indexFromPath(link.inSockPath);
        }

        if (!inSock.isValid() || !outSock.isValid())
        {
            continue;
        }

        QModelIndex nodeIdx = outSock.data(ROLE_NODE_IDX).toModelIndex();
        if (!nodeIdx.isValid())
        {
            zeno::log_warn("cannot pull node index from outSock");
            continue;
        }
        QModelIndex subgIdx = nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
        ZASSERT_EXIT(subgIdx.isValid());
        m_pModel->addLegacyLink(subgIdx, outSock, inSock);
    }
}

void ModelAcceptor::EndSubgraph()
{
    if (!m_currentGraph)
        return;
    m_currentGraph->onModelInited();
    if (m_bImport)
    {
        NODE_DESC desc;
        QString name = m_currentGraph->name();
        m_pModel->getDescriptor(name, desc);
        QModelIndexList subgNodes = m_pModel->findSubgraphNode(name);
        NodeParamModel* pNodeParamModel = nullptr;
        for (const QModelIndex& subgNode : subgNodes)
        {
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
            if (!nodeParams)
                continue;
            int row = 0;
            for (const auto& input : desc.inputs)
            {
                const QModelIndex &index = nodeParams->getParam(PARAM_INPUT, input.second.info.name);
                QVariant val;
                if (index.isValid())
                    val = index.data(ROLE_PARAM_VALUE);
                else
                    val = input.second.info.defaultValue;
                nodeParams->setAddParam(
                    PARAM_INPUT,
                    input.second.info.name,
                    input.second.info.type,
                    val,
                    input.second.info.control,
                    input.second.info.ctrlProps,
                    SOCKPROP_NORMAL,
                    DICTPANEL_INFO(),
                    input.second.info.toolTip
                );
                int srcRow = 0;
                VParamItem *pGroup = nodeParams->getInputs();
                if (pGroup)
                {
                    pGroup->getItem(input.second.info.name, &srcRow);
                    if (srcRow != row)
                    {
                        nodeParams->moveRow(pGroup->index(), srcRow, pGroup->index(), row);
                    }
                }
                row++;
            }
            row = 0;
            for (const auto& output : desc.outputs)
            {
                const QModelIndex& index = nodeParams->getParam(PARAM_OUTPUT, output.second.info.name);
                QVariant val;
                if (index.isValid())
                    val = index.data(ROLE_PARAM_VALUE);
                else
                    val = output.second.info.defaultValue;
                nodeParams->setAddParam(
                    PARAM_OUTPUT,
                    output.second.info.name,
                    output.second.info.type,
                    val,
                    output.second.info.control,
                    output.second.info.ctrlProps,
                    SOCKPROP_NORMAL,
                    DICTPANEL_INFO(),
                    output.second.info.toolTip
                );
                int srcRow = 0;
                VParamItem* pGroup = nodeParams->getOutputs();
                if (pGroup)
                {
                    pGroup->getItem(output.second.info.name, &srcRow);
                    if (srcRow != row)
                    {
                        nodeParams->moveRow(pGroup->index(), srcRow, pGroup->index(), row);
                    }
                }
                row++;
            }
            row = 0;
            for (const auto& param : desc.params)
            {
                const QModelIndex& index = nodeParams->getParam(PARAM_PARAM, param.name);
                QVariant val;
                if (index.isValid())
                    val = index.data(ROLE_PARAM_VALUE);
                else
                    val = param.defaultValue;
                nodeParams->setAddParam(
                    PARAM_PARAM,
                    param.name,
                    param.typeDesc,
                    val,
                    param.control,
                    param.controlProps,
                    SOCKPROP_NORMAL,
                    DICTPANEL_INFO(),
                    param.toolTip
                );
                int srcRow = 0;
                VParamItem* pGroup = nodeParams->getParams();
                if (pGroup)
                {
                    pGroup->getItem(param.name, &srcRow);
                    if (srcRow != row)
                    {
                        nodeParams->moveRow(pGroup->index(), srcRow, pGroup->index(), row);
                    }
                }
                row++;
            }
            INPUT_SOCKETS inputs;
            nodeParams->getInputSockets(inputs);
            for (const auto& input : inputs)
            {
                QString name = input.key();
                if (!desc.inputs.contains(name))
                {
                    nodeParams->removeParam(PARAM_INPUT, name);
                }
            }
            OUTPUT_SOCKETS outputs;
            nodeParams->getOutputSockets(outputs);
            for (const auto& output : outputs)
            {
                if (!desc.outputs.contains(output.key()))
                {
                    nodeParams->removeParam(PARAM_OUTPUT, output.key());
                }
            }
        }

    }
    m_currentGraph = nullptr;
}

void ModelAcceptor::setFilePath(const QString& fileName)
{
    if (!m_bImport)
        m_pModel->setFilePath(fileName);
}

void ModelAcceptor::switchSubGraph(const QString& graphName)
{
    m_pModel->switchSubGraph(graphName);
}

bool ModelAcceptor::addNode(QString& nodeid, const QString& name, const QString& customName, const NODE_DESCS& legacyDescs)
{
    if (!m_currentGraph)
        return false;

    bool bUnRevision = !m_pModel->hasDescriptor(name);
    /*
    if (!m_pModel->hasDescriptor(name)) {
        zeno::log_warn("no node class named [{}]", name.toStdString());
        return false;
    }
    */


    NODE_DATA data;
    if (m_bImport)
    {
        QString newId = UiHelper::generateUuid(name);
        m_oldToNewNodeIds[newId] = nodeid;
        nodeid = newId;
    }
    data[ROLE_OBJID] = nodeid;
    data[ROLE_OBJNAME] = name;
    data[ROLE_CUSTOM_OBJNAME] = customName;
    data[ROLE_COLLASPED] = false;
    data[ROLE_NODETYPE] = bUnRevision ? NO_VERSION_NODE : UiHelper::nodeType(name);

    //zeno::log_warn("zsg has Inputs {}", data.find(ROLE_PARAMETERS) != data.end());
    m_currentGraph->appendItem(data, false);
    return true;
}

void ModelAcceptor::setViewRect(const QRectF& rc)
{
    if (!m_currentGraph)
        return;
    m_currentGraph->setViewRect(rc);
}

void ModelAcceptor::setSocketKeys(const QString& id, const QStringList& keys)
{
    if (!m_currentGraph)
        return;

    //legacy io formats.

    //there is no info about whether the key belongs to input or output.
    //have to classify by nodecls.
    QModelIndex idx = m_currentGraph->index(id);
    const QString& nodeName = idx.data(ROLE_OBJNAME).toString();
    if (nodeName == "MakeDict")
    {
        for (auto keyName : keys)
        {
            addDictKey(id, keyName, true);
        }
    }
    else if (nodeName == "ExtractDict")
    {
        for (auto keyName : keys)
        {
            addDictKey(id, keyName, false);
        }
    }
    else if (nodeName == "MakeList")
    {
        //no need to do anything, because we have import the keys from inputs directly.
    }
}

void ModelAcceptor::addSocket(bool bInput, const QString& ident, const QString& sockName, const QString& sockProperty)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(ident);
    const QString& nodeCls = idx.data(ROLE_OBJNAME).toString();

    PARAM_CONTROL ctrl = CONTROL_NONE;
    SOCKET_PROPERTY prop = SOCKPROP_NORMAL;
    if (sockProperty == "dict-panel")
        prop = SOCKPROP_DICTLIST_PANEL;
    else if (sockProperty == "editable")
        prop = SOCKPROP_EDITABLE;
    else if (sockProperty == "group-line")
        prop = SOCKPROP_GROUP_LINE;

    //the layout should be standard inputs desc by latest descriptors.
    //so, we can only add dynamic key. for example, list and dict node.
    if (prop == SOCKPROP_EDITABLE || nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict")
    {
        NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
        if (prop == SOCKPROP_EDITABLE)
        {
            nodeParams->setAddParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName, "string", "", CONTROL_NONE, QVariant(), prop);
        }
        else
        {
            nodeParams->setAddParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName, "", QVariant(), CONTROL_NONE, QVariant(), prop);
        }
    }
}

void ModelAcceptor::addDictKey(const QString& id, const QString& keyName, bool bInput)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(id);
    m_currentGraph->setParamValue(bInput ? PARAM_INPUT : PARAM_OUTPUT, idx, keyName, QVariant(), "", CONTROL_NONE, SOCKPROP_EDITABLE);
}

void ModelAcceptor::initSockets(const QString& id, const QString& name, const NODE_DESCS& legacyDescs)
{
    if (!m_currentGraph)
        return;

    NODE_DESC desc;
    bool coreDesc = m_pModel->getDescriptor(name, desc);
    if (!coreDesc) {
        ZASSERT_EXIT(legacyDescs.find(name) != legacyDescs.end());
        desc = legacyDescs[name];
    }

    //params
    INPUT_SOCKETS inputs;
    PARAMS_INFO params;
    OUTPUT_SOCKETS outputs;

    for (PARAM_INFO descParam : desc.params)
    {
        PARAM_INFO param;
        param.name = descParam.name;
        param.control = descParam.control;
        param.controlProps = descParam.controlProps;
        param.typeDesc = descParam.typeDesc;
        param.defaultValue = descParam.defaultValue;
        param.value = descParam.value;
        if (!coreDesc)
            param.sockProp = SOCKPROP_LEGACY;
        params.insert(param.name, param);
    }

    for (INPUT_SOCKET descInput : desc.inputs)
    {
        INPUT_SOCKET input;
        input.info.nodeid = id;
        input.info.control = descInput.info.control;
        input.info.ctrlProps = descInput.info.ctrlProps;
        input.info.type = descInput.info.type;
        input.info.name = descInput.info.name;
        input.info.defaultValue = descInput.info.defaultValue;
        if (!coreDesc)
            input.info.sockProp = SOCKPROP_LEGACY;
        inputs.insert(input.info.name, input);
    }

    for (OUTPUT_SOCKET descOutput : desc.outputs)
    {
        OUTPUT_SOCKET output;
        output.info.nodeid = id;
        output.info.control = descOutput.info.control;
        output.info.ctrlProps = descOutput.info.ctrlProps;
        output.info.type = descOutput.info.type;
        output.info.name = descOutput.info.name;
        if (!coreDesc)
            output.info.sockProp = SOCKPROP_LEGACY;
        outputs[output.info.name] = output;
    }

    QModelIndex idx = m_currentGraph->index(id);

    m_currentGraph->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
    m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
    m_currentGraph->setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
}

void ModelAcceptor::setInputSocket(
        const QString& nodeCls,
        const QString& inNode,
        const QString& inSock,
        const QString& outNode,
        const QString& outSock,
        const rapidjson::Value& defaultValue,
        const NODE_DESCS& legacyDescs
)
{
    const QString &subgName = m_currentGraph->name();
    QString outLinkPath;
    if (!outNode.isEmpty() && !outSock.isEmpty()) {
        outLinkPath = UiHelper::constructObjPath(subgName, outNode, "[node]/outputs/", outSock);
    }
    setInputSocket2(nodeCls, inNode, inSock, outLinkPath, "", defaultValue, legacyDescs);
}

void ModelAcceptor::setInputSocket2(
                const QString& nodeCls,
                const QString& inNode,
                const QString& inSock,
                const QString& outLinkPath,
                const QString& sockProperty,
                const rapidjson::Value& defaultVal,
                const NODE_DESCS& legacyDescs)
{
    if (!m_currentGraph)
        return;

    NODE_DESC desc;
    bool isCoreDesc = m_pModel->getDescriptor(nodeCls, desc);
    if (!isCoreDesc) {
        ZASSERT_EXIT(legacyDescs.find(nodeCls) != legacyDescs.end());
        desc = legacyDescs[nodeCls];
    }

    //parse default value.
    QVariant defaultValue;
    if (!defaultVal.IsNull())
    {
        SOCKET_INFO descInfo;
        if (desc.inputs.find(inSock) != desc.inputs.end()) {
            descInfo = desc.inputs[inSock].info;
        }
        QString type = descInfo.type;
        if (defaultVal.IsObject() && defaultVal.HasMember("objectType")) {
            type = defaultVal["objectType"].GetString();
        }
        defaultValue = UiHelper::parseJsonByType(type, defaultVal, m_currentGraph);
    }

    QString subgName, paramCls;
    subgName = m_currentGraph->name();
    QString inSockPath = UiHelper::constructObjPath(subgName, inNode, "[node]/inputs/", inSock);

    QModelIndex sockIdx = m_pModel->indexFromPath(inSockPath);
    if (sockIdx.isValid())
    {
        if (!defaultValue.isNull())
        {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(sockIdx.model());
            ZASSERT_EXIT(pModel);
            pModel->setData(sockIdx, defaultValue, ROLE_PARAM_VALUE);
        }
        if (!outLinkPath.isEmpty())
        {
            //collect edge, because output socket may be not initialized.
            EdgeInfo fullLink(outLinkPath, inSockPath);
            if (isCoreDesc)
                m_subgLinks.append(fullLink);
            else
                m_subgLegacyLinks.append(fullLink);
        }
    }
    else
    {
        QModelIndex inNodeIdx = m_currentGraph->index(inNode);
        ZASSERT_EXIT(inNodeIdx.isValid());
        //the layout should be standard inputs desc by latest descriptors.
        NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(inNodeIdx.data(ROLE_NODE_PARAMS));
        ZASSERT_EXIT(nodeParams);

        //Dynamic socket
        if (nodeCls == "MakeList" || nodeCls == "MakeDict")
        {
            const QString& sockName = inSock;
            PARAM_CONTROL ctrl = CONTROL_NONE;
            SOCKET_PROPERTY prop = SOCKPROP_NORMAL;
            if (nodeCls == "MakeDict")
            {
                prop = SOCKPROP_EDITABLE;
            }
            if (!outLinkPath.isEmpty())
            {
                EdgeInfo fullLink(outLinkPath, inSockPath);
                m_subgLinks.append(fullLink);
            }

            if (prop == SOCKPROP_EDITABLE) {
                nodeParams->setAddParam(PARAM_INPUT, sockName, "string", "", CONTROL_NONE, QVariant(), prop);
            } else {
                nodeParams->setAddParam(PARAM_INPUT, sockName, "", QVariant(), CONTROL_NONE,  QVariant(), prop);
            }
        }
        else
        {
            NODE_DESC legacyDesc = legacyDescs[nodeCls];
            if (legacyDesc.inputs.find(inSock) == legacyDesc.inputs.end())
            {
                return;
            }
            SOCKET_INFO& info = legacyDesc.inputs[inSock].info;
            nodeParams->setAddParam(PARAM_LEGACY_INPUT, inSock, info.type, info.defaultValue, info.control, QVariant(), SOCKPROP_LEGACY);
            sockIdx = nodeParams->getParam(PARAM_LEGACY_INPUT, inSock);
            ZASSERT_EXIT(sockIdx.isValid());
            inSockPath = sockIdx.data(ROLE_OBJPATH).toString();
            if (!outLinkPath.isEmpty())
            {
                //collect edge, because output socket may be not initialized.
                EdgeInfo fullLink(outLinkPath, inSockPath);
                m_subgLegacyLinks.append(fullLink);
            }
            zeno::log_warn("{}: input socket {} is at legacy version", nodeCls.toStdString(), inSock.toStdString());
        }
    }
}

void ModelAcceptor::setOutputSocket(const QString& inNode, const QString& inSock, const QString& netlabel, const QString& type)
{
    if (!m_currentGraph)
        return;
    QString subgName;
    subgName = m_currentGraph->name();
    QString inSockPath = UiHelper::constructObjPath(subgName, inNode, "[node]/outputs/", inSock);
    QModelIndex sockIdx = m_pModel->indexFromPath(inSockPath);
    ZASSERT_EXIT(sockIdx.isValid());
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(sockIdx.model());
    ZASSERT_EXIT(pModel);
    if (!netlabel.isEmpty())
    {
        m_currentGraph->addNetLabel(sockIdx, netlabel, false);
    }
    if (!type.isEmpty())
    {
        pModel->setData(sockIdx, type, ROLE_PARAM_TYPE);
    }
}

void ModelAcceptor::setDictPanelProperty(bool bInput, const QString& ident, const QString& sockName, bool bCollasped)
{
    QModelIndex inNodeIdx = m_currentGraph->index(ident);
    ZASSERT_EXIT(inNodeIdx.isValid());

    QModelIndex sockIdx = m_currentGraph->nodeParamIndex(inNodeIdx, bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);
    ZASSERT_EXIT(sockIdx.isValid());

    DictKeyModel *keyModel = QVariantPtr<DictKeyModel>::asPtr(sockIdx.data(ROLE_VPARAM_LINK_MODEL));
    if (!keyModel) {
        return;
    }
    //ZERROR_EXIT(keyModel);
    keyModel->setCollasped(bCollasped);
}

void ModelAcceptor::setControlAndProperties(const QString& nodeCls, const QString& inNode, const QString& inSock, PARAM_CONTROL control, const QVariant& ctrlProperties) 
{
    //init control by descriptor
    return;
    if (!m_currentGraph)
        return;
    if (m_pModel->hasDescriptor(nodeCls))
        return;
    QString subgName, paramCls;
    subgName = m_currentGraph->name();
    QString inSockPath = UiHelper::constructObjPath(subgName, inNode, "[node]/inputs/", inSock);

    QModelIndex sockIdx = m_pModel->indexFromPath(inSockPath);
    if (sockIdx.isValid()) {
        QAbstractItemModel *pModel = const_cast<QAbstractItemModel *>(sockIdx.model());
        ZASSERT_EXIT(pModel);
        pModel->setData(sockIdx, control, ROLE_PARAM_CTRL);
        pModel->setData(sockIdx, ctrlProperties, ROLE_VPARAM_CTRL_PROPERTIES);
    } else {
         zeno::log_warn("{}: no such input socket {}", nodeCls.toStdString(), inSock.toStdString());
    }
}

void ModelAcceptor::setNetLabel(PARAM_CLASS cls, const QString& inNode, const QString& inSock, const QString& netlabel)
{
    if (!m_currentGraph)
        return;
    QString nodeCls;
    if (cls == PARAM_INPUT)
        nodeCls = "inputs";
    else if (cls == PARAM_OUTPUT)
        nodeCls = "outputs";
    QString inSockPath = UiHelper::constructObjPath(m_currentGraph->name(), inNode, QString("[node]/%1/").arg(nodeCls), inSock);
    QModelIndex sockIdx = m_pModel->indexFromPath(inSockPath);
    ZASSERT_EXIT(sockIdx.isValid());
    m_currentGraph->addNetLabel(sockIdx, netlabel, cls == PARAM_INPUT);
}

void ModelAcceptor::setToolTip(PARAM_CLASS cls, const QString &inNode, const QString &inSock, const QString &toolTip) 
{
    if (!m_currentGraph)
         return;
    QString nodeCls;
    if (cls == PARAM_INPUT)
         nodeCls = "inputs";
    else if (cls == PARAM_OUTPUT)
         nodeCls = "outputs";
    QString inSockPath = UiHelper::constructObjPath(m_currentGraph->name(), inNode, QString("[node]/%1/").arg(nodeCls), inSock);
    QModelIndex sockIdx = m_pModel->indexFromPath(inSockPath);
    ZASSERT_EXIT(sockIdx.isValid());
    QAbstractItemModel *pModel = const_cast<QAbstractItemModel *>(sockIdx.model());
    ZASSERT_EXIT(pModel);
    pModel->setData(sockIdx, toolTip, ROLE_VPARAM_TOOLTIP);
}
void ModelAcceptor::addInnerDictKey(
            bool bInput,
            const QString& ident,
            const QString& sockName,
            const QString& keyName,
            const QString& link,
            const QString& netLabel
            )
{
    QModelIndex inNodeIdx = m_currentGraph->index(ident);
    ZASSERT_EXIT(inNodeIdx.isValid());

    QModelIndex sockIdx = m_currentGraph->nodeParamIndex(inNodeIdx, bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);
    ZASSERT_EXIT(sockIdx.isValid());

    DictKeyModel* keyModel = QVariantPtr<DictKeyModel>::asPtr(sockIdx.data(ROLE_VPARAM_LINK_MODEL));
    int n = keyModel->rowCount();
    keyModel->insertRow(n);
    const QModelIndex& newKeyIdx = keyModel->index(n, 0);
    ZASSERT_EXIT(newKeyIdx.isValid());
    keyModel->setData(newKeyIdx, keyName, ROLE_PARAM_NAME);

    if (bInput && !link.isEmpty())
    {
        QString keySockPath = newKeyIdx.data(ROLE_OBJPATH).toString();
        EdgeInfo fullLink(link, keySockPath);
        m_subgLinks.append(fullLink);
    }
    if (!netLabel.isEmpty())
    {
        m_currentGraph->addNetLabel(newKeyIdx, netLabel, bInput);
    }
}

void ModelAcceptor::endInputs(const QString& id, const QString& nodeCls)
{
    //todo
}

void ModelAcceptor::addCustomUI(const QString& id, const VPARAM_INFO& invisibleRoot)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(id);
    m_currentGraph->setData(idx, QVariant::fromValue(invisibleRoot), ROLE_CUSTOMUI_PANEL_IO);
}

void ModelAcceptor::setIOVersion(zenoio::ZSG_VERSION versio)
{
    m_pModel->setIOVersion(versio);
}

void ModelAcceptor::endNode(const QString& id, const QString& nodeCls, const rapidjson::Value& objValue)
{
    if (objValue.HasMember("outputs"))
    {
        const rapidjson::Value& outputs = objValue["outputs"];
        for (const auto& outObj : outputs.GetObject())
        {
            const QString& outSock = outObj.name.GetString();
            const auto& sockObj = outObj.value;
            if (sockObj.IsObject())
            {
                if (sockObj.HasMember("tooltip")) {
                    QString toolTip = QString::fromUtf8(sockObj["tooltip"].GetString());
                    setToolTip(PARAM_OUTPUT, id, outSock, toolTip);
                }
                QString netlabel;
                if (sockObj.HasMember("netlabel"))
                {
                    netlabel = QString::fromUtf8(sockObj["netlabel"].GetString());
                }
                QString type;
                if (sockObj.HasMember("type"))
                {
                    type = sockObj["type"].GetString();
                }
                if (!netlabel.isEmpty() || !type.isEmpty())
                    setOutputSocket(id, outSock, netlabel, type);
            }
        }
    }
}

void ModelAcceptor::addCommandParam(const rapidjson::Value& val, const QString& path)
{
    QModelIndex sockIdx = m_pModel->indexFromPath(path);
    if (sockIdx.isValid())
    {
        CommandParam param;
        if (val.HasMember("name"))
            param.name = val["name"].GetString();
        if (val.HasMember("description"))
            param.description = val["description"].GetString();
        param.value = sockIdx.data(ROLE_PARAM_VALUE);
        m_pModel->addCommandParam(path, param);
    }
}

void ModelAcceptor::endParams(const QString& id, const QString& nodeCls)
{
    if (nodeCls == "SubInput" || nodeCls == "SubOutput")
    {
        const QModelIndex& idx = m_currentGraph->index(id);

        const QModelIndex &nameIdx = m_currentGraph->nodeParamIndex(idx, PARAM_PARAM, "name");
        const QModelIndex &typeIdx = m_currentGraph->nodeParamIndex(idx, PARAM_PARAM, "type");
        const QModelIndex &deflIdx = m_currentGraph->nodeParamIndex(idx, PARAM_PARAM, "defl");

        ZASSERT_EXIT(nameIdx.isValid() && typeIdx.isValid() && deflIdx.isValid());

        const QString& type = typeIdx.data(ROLE_PARAM_VALUE).toString();
        QVariant deflVal = deflIdx.data(ROLE_PARAM_VALUE).toString();
        deflVal = UiHelper::parseVarByType(type, deflVal, nullptr);
        PARAM_CONTROL control = UiHelper::getControlByType(type);
        m_currentGraph->setParamValue(PARAM_PARAM, idx, "defl", deflVal, type, control);
    }
}

void ModelAcceptor::setParamValue(const QString& id, const QString& nodeCls, const QString& name, const rapidjson::Value& value, const NODE_DESCS& legacyDescs)
{
    if (!m_currentGraph)
        return;

    NODE_DESC desc;
    bool isCoreDesc = m_pModel->getDescriptor(nodeCls, desc);
    if (!isCoreDesc) {
        ZASSERT_EXIT(legacyDescs.find(nodeCls) != legacyDescs.end());
        desc = legacyDescs[nodeCls];
    }

    QVariant var;
    if (!value.IsNull())
    {
        PARAM_INFO paramInfo;
        if (desc.params.find(name) != desc.params.end()) {
            paramInfo = desc.params[name];
        }
        if (nodeCls == "SubInput" || nodeCls == "SubOutput")
            var = UiHelper::parseJsonByValue(paramInfo.typeDesc, value, nullptr);   //dynamic type on SubInput defl.
        else
            var = UiHelper::parseJsonByType(paramInfo.typeDesc, value, m_currentGraph);
    }

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());
    PARAMS_INFO params = m_currentGraph->data(idx, ROLE_PARAMETERS).value<PARAMS_INFO>();

    if (params.find(name) != params.end())
    {
        zeno::log_trace("found param name {}", name.toStdString());
        params[name].value = var;
        m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
    }
    else
    {
        PARAMS_INFO _params = m_currentGraph->data(idx, ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
        _params[name].value = var;
        m_currentGraph->setData(idx, QVariant::fromValue(_params), ROLE_PARAMS_NO_DESC);

		if (name == "_KEYS" && (
			nodeCls == "MakeDict" ||
			nodeCls == "ExtractDict" ||
			nodeCls == "MakeList"))
		{
			//parse by socket_keys in zeno2.
			return;
		}
        if (nodeCls == "MakeCurvemap" && (name == "_POINTS" || name == "_HANDLERS"))
        {
            PARAM_INFO paramData;
            paramData.control = CONTROL_NONVISIBLE;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            params[name] = paramData;
            m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
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
            m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
            return;
        }
        if (nodeCls == "DynamicNumber" && (name == "_CONTROL_POINTS" || name == "_TMP"))
        {
            PARAM_INFO paramData;
            paramData.control = CONTROL_NONVISIBLE;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            params[name] = paramData;
            m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
            return;
        }
        zeno::log_warn("not found param name {}", name.toStdString());
    }
}

void ModelAcceptor::setParamValue2(const QString &id, const QString &noCls, const PARAMS_INFO &params) 
{
    if (!m_currentGraph)
        return;
    if (params.isEmpty())
        return;

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());

    m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
}

void ModelAcceptor::setPos(const QString& id, const QPointF& pos)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());
    m_currentGraph->setData(idx, pos, ROLE_OBJPOS);
}

void ModelAcceptor::setOptions(const QString& id, const QStringList& options)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());
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
        else if (optName == "CACHE")
        {
            opts |= OPT_CACHE;
        }
        else if (optName == "collapsed")
        {
            m_currentGraph->setData(idx, true, ROLE_COLLASPED);
        }
    }
    m_currentGraph->setData(idx, opts, ROLE_OPTIONS);
}

void ModelAcceptor::setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps)
{
    /* keep legacy format
    if (!m_currentGraph)
        return;

    QLinearGradient linearGrad;
    for (COLOR_RAMP ramp : colorRamps)
    {
        linearGrad.setColorAt(ramp.pos, QColor::fromRgbF(ramp.r, ramp.g, ramp.b));
    }

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());

    PARAMS_INFO params = m_currentGraph->data(idx, ROLE_PARAMETERS).value<PARAMS_INFO>();

    PARAM_INFO param;
    param.name = "color";
    param.control = CONTROL_COLOR;
    param.value = QVariant::fromValue(linearGrad);
    params.insert(param.name, param);
    m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
    */
}

void ModelAcceptor::setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());
    m_pModel->updateBlackboard(id, QVariant::fromValue(blackboard), m_pModel->indexBySubModel(m_currentGraph), false);
}

void ModelAcceptor::setLegacyCurve(
                    const QString& id,
                    const QVector<QPointF>& pts,
                    const QVector<QPair<QPointF, QPointF>>& hdls)
{
    if (!m_currentGraph)
        return;

    QModelIndex idx = m_currentGraph->index(id);
    ZASSERT_EXIT(idx.isValid());

    //no id in legacy curvemap.
    //todo: enable editting of legacy curve.
    //only need to parse _POINTS and __HANDLES to core legacy node like Makecurvemap.
}

QObject* ModelAcceptor::currGraphObj()
{
    return m_currentGraph;
}
