#include <QObject>
#include <QtWidgets>
#include <rapidjson/document.h>
#include "treeacceptor.h"
#include "graphstreemodel.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include <zeno/utils/logger.h>
#include "magic_enum.hpp"
#include "zassert.h"
#include "uihelper.h"
#include "variantptr.h"
#include "dictkeymodel.h"
#include "graphsmanagment.h"


TreeAcceptor::TreeAcceptor(GraphsTreeModel* pModel, GraphsModel* pSubgraphs, bool bImport)
    : m_pNodeModel(pModel)
    , m_pSubgraphs(pSubgraphs)
    , m_bImport(bImport)
    , m_pSubgAcceptor(std::make_shared<ModelAcceptor>(m_pSubgraphs, bImport))
    , m_ioVer(zenoio::VER_3)
{
    ZASSERT_EXIT(m_pNodeModel && m_pSubgraphs);
    m_pNodeModel->initSubgraphs(m_pSubgraphs);
}


QModelIndex TreeAcceptor::_getNodeIdx(const QString& identOrObjPath)
{
    QModelIndex nodeIdx = m_pNodeModel->indexFromPath(identOrObjPath);
    if (nodeIdx.isValid())
    {
        return nodeIdx;
    }
    else
    {
        return m_pNodeModel->index(identOrObjPath, m_pNodeModel->mainIndex());
    }
}

QModelIndex TreeAcceptor::_getSockIdx(const QString& inNode, const QString& sockName, bool bInput)
{
    QModelIndex nodeIdx = m_pNodeModel->indexFromPath(inNode);
    if (!nodeIdx.isValid())
    {
        nodeIdx = m_pNodeModel->index(inNode, m_pNodeModel->mainIndex());
    }
    if (!nodeIdx.isValid())
        return QModelIndex();

    QString path = nodeIdx.data(ROLE_OBJPATH).toString();
    if (bInput)
        path += cPathSeperator + QString("[node]/inputs/") + sockName;
    else
        path += cPathSeperator + QString("[node]/outputs/") + sockName;
    return m_pNodeModel->indexFromPath(path);
}

//IAcceptor
bool TreeAcceptor::setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS& nodesParams)
{
    return m_pSubgAcceptor->setLegacyDescs(graphObj, nodesParams);
}

void TreeAcceptor::BeginSubgraph(const QString& name)
{
    //must parse subgraph first.
    m_bImportMain = name == "main";
    if (!m_bImportMain)
    {
        m_pSubgAcceptor->BeginSubgraph(name);
    }
    else
    {
        m_pSubgAcceptor->resolveAllLinks();
    }
}

bool TreeAcceptor::setCurrentSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx)
{
    return false;
}

void TreeAcceptor::EndSubgraph()
{
    if (!m_bImportMain) {
        m_pSubgAcceptor->EndSubgraph();
    }
}

void TreeAcceptor::EndGraphs()
{
    resolveAllLinks();
}

void TreeAcceptor::resolveAllLinks()
{
    //add links on this subgraph.
    for (EdgeInfo link : m_links)
    {
        QModelIndex inSock, outSock, inNode, outNode;
        QString subgName, inNodeCls, outNodeCls, inSockName, outSockName, paramCls;

        if (!link.outSockPath.isEmpty())
        {
            outSock = m_pNodeModel->indexFromPath(link.outSockPath);
            outSockName = link.outSockPath;
        }
        if (!link.inSockPath.isEmpty())
        {
            inSock = m_pNodeModel->indexFromPath(link.inSockPath);
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
        m_pNodeModel->addLink(subgIdx, outSock, inSock);
    }
}

void TreeAcceptor::switchSubGraph(const QString &graphName)
{
    //todo: deprecated.
}

bool TreeAcceptor::addNode(const QString &nodeid, const QString &name, const QString &customName,
             const NODE_DESCS &descriptors)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->addNode(nodeid, name, customName, descriptors);

    //is current node a subgraph node?
    SubGraphModel* pSubgraph = m_pSubgraphs->subGraph(name);
    if (pSubgraph && zenoio::VER_3 != m_ioVer)
    {
        //recursivly fork.
        //input: 
        // 1.GraphsModel
        // 2.to fork name: name
        // 3.nodeid and customname
        QList<EdgeInfo> links = m_pNodeModel->addSubnetNode(m_pSubgraphs, name, nodeid, customName);
        m_links.append(links);
        //output:
        // links.
    }
    else
    {
        NODE_DESC desc;
        auto &inst = GraphsManagment::instance();
        if (!inst.getDescriptor(name, desc)) {
            zeno::log_warn("no node class named [{}]", name.toStdString());
            return false;
        }

        NODE_DATA data;
        data.ident = nodeid;
        data.nodeCls = name;
        data.customName = customName;
        data.bCollasped = false;
        data.type = inst.nodeType(name);
        m_pNodeModel->addNode(data, m_pNodeModel->mainIndex());
    }
    return true;
}

void TreeAcceptor::setViewRect(const QRectF &rc)
{

}

void TreeAcceptor::setSocketKeys(const QString &id, const QStringList &keys)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->setSocketKeys(id, keys);

    //legacy io formats.

    //there is no info about whether the key belongs to input or output.
    //have to classify by nodecls.
    QModelIndex idx = _getNodeIdx(id);
    const QString &nodeName = idx.data(ROLE_OBJNAME).toString();
    if (nodeName == "MakeDict") {
        for (auto keyName : keys) {
            addDictKey(id, keyName, true);
        }
    } else if (nodeName == "ExtractDict") {
        for (auto keyName : keys) {
            addDictKey(id, keyName, false);
        }
    } else if (nodeName == "MakeList") {
        //no need to do anything, because we have import the keys from inputs directly.
    }
}

void TreeAcceptor::initSockets(const QString &nodePath, const QString &name, const NODE_DESCS &descs)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->initSockets(nodePath, name, descs);

    NODE_DESC desc;
    auto &mgr = GraphsManagment::instance();
    bool ret = mgr.getDescriptor(name, desc);
    ZASSERT_EXIT(ret);

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
        params.insert(param.name, param);
    }

    for (INPUT_SOCKET descInput : desc.inputs)
    {
        INPUT_SOCKET input;
        input.info.control = descInput.info.control;
        input.info.ctrlProps = descInput.info.ctrlProps;
        input.info.type = descInput.info.type;
        input.info.name = descInput.info.name;
        input.info.defaultValue = descInput.info.defaultValue;
        inputs.insert(input.info.name, input);
    }

    for (OUTPUT_SOCKET descOutput : desc.outputs)
    {
        OUTPUT_SOCKET output;
        output.info.control = descOutput.info.control;
        output.info.ctrlProps = descOutput.info.ctrlProps;
        output.info.type = descOutput.info.type;
        output.info.name = descOutput.info.name;
        outputs[output.info.name] = output;
    }

    QModelIndex idx = _getNodeIdx(nodePath);
    ZASSERT_EXIT(idx.isValid());

    m_pNodeModel->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
    m_pNodeModel->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
    m_pNodeModel->setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
}

void TreeAcceptor::addDictKey(const QString &nodePath, const QString &keyName, bool bInput)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->addDictKey(nodePath, keyName, bInput);

    QModelIndex idx = _getNodeIdx(nodePath);
    ZASSERT_EXIT(idx.isValid());

    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(nodeParams);

    nodeParams->setAddParam(bInput ? PARAM_INPUT : PARAM_OUTPUT,
                            keyName,
                            "",                 //socket type
                            QVariant(),         //defl value
                            CONTROL_NONE,
                            QVariant(),         //ctrl props
                            SOCKPROP_EDITABLE);
}

void TreeAcceptor::addSocket(bool bInput, const QString &nodePath, const QString &sockName,
                             const QString &sockProperty)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->addSocket(bInput, nodePath, sockName, sockProperty);

    QModelIndex nodeIdx = _getNodeIdx(nodePath);
    ZASSERT_EXIT(nodeIdx.isValid());

    const QString &nodeCls = nodeIdx.data(ROLE_OBJNAME).toString();

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
    if (prop == SOCKPROP_EDITABLE || 
        nodeCls == "MakeList" ||
        nodeCls == "MakeDict" ||
        nodeCls == "ExtractDict")
    {
        NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
        ZASSERT_EXIT(nodeParams);
        if (prop == SOCKPROP_EDITABLE) {
            nodeParams->setAddParam(bInput ? PARAM_INPUT : PARAM_OUTPUT,
                                    sockName,
                                    "string",
                                    "",
                                    CONTROL_NONE,
                                    QVariant(),
                                    prop);
        } else {
            nodeParams->setAddParam(bInput ? PARAM_INPUT : PARAM_OUTPUT,
                                    sockName,
                                    "",
                                    QVariant(),
                                    CONTROL_NONE,
                                    QVariant(),
                                    prop);
        }
    }
}

void TreeAcceptor::setInputSocket2(
                    const QString &nodeCls,
                    const QString &inNode,
                    const QString &inSock,
                    const QString &outLinkPath,
                    const QString &sockProperty,
                    const rapidjson::Value &defaultVal)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setInputSocket2(nodeCls, inNode, inSock, outLinkPath, sockProperty, defaultVal);
    }

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

    //inNode may be a obj path or ident on main subgraph.
    QModelIndex inNodeIdx = _getNodeIdx(inNode);
    ZASSERT_EXIT(inNodeIdx.isValid());
    const QString& objPath = inNodeIdx.data(ROLE_OBJPATH).toString();
    QString inSockPath = objPath + ":[node]/inputs/" + inSock;

    QModelIndex sockIdx = m_pNodeModel->indexFromPath(inSockPath);
    if (sockIdx.isValid())
    {
        if (!defaultValue.isNull())
        {
            QAbstractItemModel *pModel = const_cast<QAbstractItemModel *>(sockIdx.model());
            ZASSERT_EXIT(pModel);
            pModel->setData(sockIdx, defaultValue, ROLE_PARAM_VALUE);
        }
        if (!outLinkPath.isEmpty())
        {
            //collect edge, because output socket may be not initialized.
            EdgeInfo fullLink(outLinkPath, inSockPath);
            m_links.append(fullLink);
        }
    }
    else
    {
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
                m_links.append(fullLink);
            }

            //the layout should be standard inputs desc by latest descriptors.
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(inNodeIdx.data(ROLE_NODE_PARAMS));
            if (prop == SOCKPROP_EDITABLE) {
                nodeParams->setAddParam(PARAM_INPUT, sockName, "string", "", CONTROL_NONE, QVariant(), prop);
            } else {
                nodeParams->setAddParam(PARAM_INPUT, sockName, "", QVariant(), CONTROL_NONE,  QVariant(), prop);
            }
        }
        else
        {
            zeno::log_warn("{}: no such input socket {}", nodeCls.toStdString(), inSock.toStdString());
        }
    }
}

void TreeAcceptor::setInputSocket(
                    const QString &nodeCls,
                    const QString &inNode,
                    const QString &inSock,
                    const QString &outNode,
                    const QString &outSock,
                    const rapidjson::Value &defaultValue)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setInputSocket(nodeCls, inNode, inSock, outNode, outSock, defaultValue);
    }

    QString outLinkPath;
    if (!outNode.isEmpty() && !outSock.isEmpty())
    {
        if (outNode.indexOf('/') != -1) {
            outLinkPath = outNode + ":" + "[node]/outputs/" + outSock;
        } else {
            outLinkPath = "/main/" + outNode + ":" + "[node]/outputs/" + outSock;
        }
    }
    setInputSocket2(nodeCls, inNode, inSock, outLinkPath, "", defaultValue);
}

void TreeAcceptor::addInnerDictKey(
                    bool bInput,
                    const QString &inNode,
                    const QString &sockName,
                    const QString &keyName,
                    const QString &link)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->addInnerDictKey(bInput, inNode, sockName, keyName, link);
    }

    const QModelIndex& nodeIdx = _getNodeIdx(inNode);
    ZASSERT_EXIT(nodeIdx.isValid());

    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(nodeParams);
    QModelIndex sockIdx = nodeParams->getParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);

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
        m_links.append(fullLink);
    }
}

void TreeAcceptor::setDictPanelProperty(
                    bool bInput,
                    const QString &ident,
                    const QString &sockName,
                    bool bCollasped)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setDictPanelProperty(bInput, ident, sockName, bCollasped);
    }

    QModelIndex nodeIdx = _getNodeIdx(ident);
    ZASSERT_EXIT(nodeIdx.isValid());

    NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(nodeParams);
    QModelIndex sockIdx = nodeParams->getParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);

    DictKeyModel* keyModel = QVariantPtr<DictKeyModel>::asPtr(sockIdx.data(ROLE_VPARAM_LINK_MODEL));
    if (!keyModel) {
        return;
    }
    //ZERROR_EXIT(keyModel);
    keyModel->setCollasped(bCollasped);
}

void TreeAcceptor::setControlAndProperties(
                    const QString &nodeCls,
                    const QString &inNode,
                    const QString &inSock,
                    PARAM_CONTROL control,
                    const QVariant &ctrlProperties)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setControlAndProperties(nodeCls, inNode, inSock, control, ctrlProperties);
    }

    QModelIndex sockIdx = _getSockIdx(inNode, inSock, true);
    if (sockIdx.isValid()) {
        QAbstractItemModel *pModel = const_cast<QAbstractItemModel *>(sockIdx.model());
        ZASSERT_EXIT(pModel);
        pModel->setData(sockIdx, control, ROLE_PARAM_CTRL);
        pModel->setData(sockIdx, ctrlProperties, ROLE_VPARAM_CTRL_PROPERTIES);
    } else {
        zeno::log_warn("{}: no such input socket {}", nodeCls.toStdString(), inSock.toStdString());
    }
}

void TreeAcceptor::setToolTip(PARAM_CLASS cls, const QString &inNode, const QString &inSock,
                              const QString &toolTip)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setToolTip(cls, inNode, inSock, toolTip);
    }

    QModelIndex sockIdx = _getSockIdx(inNode, inSock, cls == PARAM_INPUT);
    ZASSERT_EXIT(sockIdx.isValid());
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(sockIdx.model());
    ZASSERT_EXIT(pModel);
    pModel->setData(sockIdx, toolTip, ROLE_VPARAM_TOOLTIP);
}

void TreeAcceptor::setParamValue(const QString &id, const QString &nodeCls, const QString &name,
                   const rapidjson::Value &value)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setParamValue(id, nodeCls, name, value);
    }

    NODE_DESC desc;
    auto &mgr = GraphsManagment::instance();
    bool ret = mgr.getDescriptor(nodeCls, desc);
    ZASSERT_EXIT(ret);

    QVariant var;
    if (!value.IsNull()) {
        PARAM_INFO paramInfo;
        if (desc.params.find(name) != desc.params.end()) {
            paramInfo = desc.params[name];
        }
        if (nodeCls == "SubInput" || nodeCls == "SubOutput")
            var = UiHelper::parseJsonByValue(paramInfo.typeDesc, value); //dynamic type on SubInput defl.
        else
            var = UiHelper::parseJsonByType(paramInfo.typeDesc, value);
    }

    QModelIndex nodeIdx = _getNodeIdx(id);  //id can be a objpath, like /main/subgA/subgB/xxx-wrangle.
    ZASSERT_EXIT(nodeIdx.isValid());
    PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();

    if (params.find(name) != params.end())
    {
        zeno::log_trace("found param name {}", name.toStdString());
        params[name].value = var;
        m_pNodeModel->setData(nodeIdx, QVariant::fromValue(params), ROLE_PARAMETERS);
    }
    else
    {
        PARAMS_INFO _params = nodeIdx.data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
        _params[name].value = var;
        m_pNodeModel->setData(nodeIdx, QVariant::fromValue(_params), ROLE_PARAMS_NO_DESC);
        if (name == "_KEYS" && (nodeCls == "MakeDict" || nodeCls == "ExtractDict" || nodeCls == "MakeList")) {
            //parse by socket_keys in zeno2.
            return;
        }
        if (nodeCls == "MakeCurvemap" && (name == "_POINTS" || name == "_HANDLERS")) {
            PARAM_INFO paramData;
            paramData.control = CONTROL_NONVISIBLE;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            params[name] = paramData;
            m_pNodeModel->setData(nodeIdx, QVariant::fromValue(params), ROLE_PARAMETERS);
            return;
        }
        if (nodeCls == "MakeHeatmap" && name == "_RAMPS") {
            PARAM_INFO paramData;
            paramData.control = CONTROL_COLOR;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            params[name] = paramData;
            m_pNodeModel->setData(nodeIdx, QVariant::fromValue(params), ROLE_PARAMETERS);
            return;
        }
        if (nodeCls == "DynamicNumber" && (name == "_CONTROL_POINTS" || name == "_TMP")) {
            PARAM_INFO paramData;
            paramData.control = CONTROL_NONVISIBLE;
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            params[name] = paramData;
            m_pNodeModel->setData(nodeIdx, QVariant::fromValue(params), ROLE_PARAMETERS);
            return;
        }
        zeno::log_warn("not found param name {}", name.toStdString());
    }
}

void TreeAcceptor::setParamValue2(const QString &id, const QString &noCls, const PARAMS_INFO &params)
{
    if (!m_bImportMain) {
        return m_pSubgAcceptor->setParamValue2(id, noCls, params);
    }

    if (params.isEmpty())
        return;

    QModelIndex idx = _getNodeIdx(id);
    ZASSERT_EXIT(idx.isValid());
    m_pNodeModel->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
}

void TreeAcceptor::setPos(const QString &id, const QPointF &pos)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->setPos(id, pos);

    QModelIndex idx = _getNodeIdx(id);
    ZASSERT_EXIT(idx.isValid());
    m_pNodeModel->setData(idx, pos, ROLE_OBJPOS);
}

void TreeAcceptor::setOptions(const QString &id, const QStringList &options)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->setOptions(id, options);

    QModelIndex idx = _getNodeIdx(id);
    ZASSERT_EXIT(idx.isValid());
    int opts = 0;
    for (int i = 0; i < options.size(); i++) {
        const QString &optName = options[i];
        if (optName == "ONCE") {
            opts |= OPT_ONCE;
        } else if (optName == "PREP") {
            opts |= OPT_PREP;
        } else if (optName == "VIEW") {
            opts |= OPT_VIEW;
        } else if (optName == "MUTE") {
            opts |= OPT_MUTE;
        } else if (optName == "collapsed") {
            m_pNodeModel->setData(idx, true, ROLE_COLLASPED);
        }
    }
    m_pNodeModel->setData(idx, opts, ROLE_OPTIONS);
}

void TreeAcceptor::setColorRamps(const QString &id, const COLOR_RAMPS &colorRamps)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->setColorRamps(id, colorRamps);

    //todo: deprecated.
}

void TreeAcceptor::setBlackboard(const QString &id, const BLACKBOARD_INFO &blackboard)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->setBlackboard(id, blackboard);

    //todo: how to specify subgIdx.
    QModelIndex idx = _getNodeIdx(id);
    ZASSERT_EXIT(idx.isValid());
    m_pNodeModel->updateBlackboard(id, QVariant::fromValue(blackboard), QModelIndex(), false);
}

void TreeAcceptor::setTimeInfo(const TIMELINE_INFO &info)
{
    m_timeInfo.beginFrame = qMin(info.beginFrame, info.endFrame);
    m_timeInfo.endFrame = qMax(info.beginFrame, info.endFrame);
    m_timeInfo.currFrame = qMax(qMin(m_timeInfo.currFrame, m_timeInfo.endFrame), m_timeInfo.beginFrame);
}

TIMELINE_INFO TreeAcceptor::timeInfo() const
{
    return m_timeInfo;
}

void TreeAcceptor::setLegacyCurve(const QString &id, const QVector<QPointF> &pts,
                    const QVector<QPair<QPointF, QPointF>> &hdls)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->setLegacyCurve(id, pts, hdls);

    //todo: deprecated.
}

QObject *TreeAcceptor::currGraphObj()
{
    return nullptr;
}

void TreeAcceptor::endInputs(const QString &id, const QString &nodeCls)
{

}

void TreeAcceptor::endParams(const QString &id, const QString &nodeCls)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->endParams(id, nodeCls);

    if (nodeCls == "SubInput" || nodeCls == "SubOutput")
    {
        const QModelIndex& idx = _getNodeIdx(id);
        NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
        ZASSERT_EXIT(nodeParams);

        const QModelIndex& nameIdx = nodeParams->getParam(PARAM_PARAM, "name");
        const QModelIndex& typeIdx = nodeParams->getParam(PARAM_PARAM, "type");
        const QModelIndex& deflIdx = nodeParams->getParam(PARAM_PARAM, "defl");

        ZASSERT_EXIT(nameIdx.isValid() && typeIdx.isValid() && deflIdx.isValid());
        const QString& type = typeIdx.data(ROLE_PARAM_VALUE).toString();
        QVariant deflVal = deflIdx.data(ROLE_PARAM_VALUE).toString();
        deflVal = UiHelper::parseVarByType(type, deflVal);
        PARAM_CONTROL control = UiHelper::getControlByType(type);

        nodeParams->setAddParam(PARAM_PARAM, "defl", type, deflVal, control, QVariant());
    }
}

void TreeAcceptor::addCustomUI(const QString &id, const VPARAM_INFO &invisibleRoot)
{
    if (!m_bImportMain)
        return m_pSubgAcceptor->addCustomUI(id, invisibleRoot);

    QModelIndex idx = _getNodeIdx(id);
    m_pNodeModel->setData(idx, QVariant::fromValue(invisibleRoot), ROLE_CUSTOMUI_PANEL_IO);
}

void TreeAcceptor::setIOVersion(zenoio::ZSG_VERSION versio)
{
    m_ioVer = versio;
    m_pSubgAcceptor->setIOVersion(versio);
    m_pNodeModel->setIOVersion(versio);
}
