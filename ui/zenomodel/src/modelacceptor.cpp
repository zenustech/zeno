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


ModelAcceptor::ModelAcceptor(GraphsModel* pModel, bool bImport)
    : m_pModel(pModel)
    , m_currentGraph(nullptr)
    , m_bImport(bImport)
{
}

bool ModelAcceptor::setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS& legacyDescs)
{
    //discard legacy desc except subnet desc.
    QStringList subgraphs;
    for (const auto& subgraph : graphObj.GetObject())
    {
        if (subgraph.name != "main") {
            subgraphs.append(QString::fromUtf8(subgraph.name.GetString()));
        }
    }
    QList<NODE_DESC> subnetDescs;
    for (QString name : subgraphs)
    {
        if (legacyDescs.find(name) == legacyDescs.end())
        {
            zeno::log_warn("subgraph {} isn't described by the file descs.", name.toStdString());
            continue;
        }
        subnetDescs.append(legacyDescs[name]);
    }
    bool ret = m_pModel->appendSubnetDescsFromZsg(subnetDescs);
    return ret;
}

void ModelAcceptor::setTimeInfo(const TIMELINE_INFO& info)
{
    m_timeInfo.beginFrame = qMin(info.beginFrame, info.endFrame);
    m_timeInfo.endFrame = qMax(info.beginFrame, info.endFrame);
    m_timeInfo.currFrame = qMax(qMin(m_timeInfo.currFrame, m_timeInfo.endFrame),
        m_timeInfo.beginFrame);
}

TIMELINE_INFO ModelAcceptor::timeInfo() const
{
    return m_timeInfo;
}

void ModelAcceptor::BeginSubgraph(const QString& name)
{
    if (m_bImport && name == "main")
    {
        m_currentGraph = nullptr;
        return;
    }

    if (m_bImport)
        zeno::log_info("Importing subgraph {}", name.toStdString());

    ZASSERT_EXIT(m_pModel && !m_currentGraph);
    SubGraphModel* pSubModel = new SubGraphModel(m_pModel);
    pSubModel->setName(name);
    m_pModel->appendSubGraph(pSubModel);
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
        m_pModel->addLink(outSock, inSock);
    }
}

void ModelAcceptor::EndSubgraph()
{
    if (!m_currentGraph)
        return;
    m_currentGraph->onModelInited();
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

bool ModelAcceptor::addNode(const QString& nodeid, const QString& name, const NODE_DESCS& legacyDescs)
{
    if (!m_currentGraph)
        return false;

    if (!m_pModel->hasDescriptor(name)) {
        zeno::log_warn("no node class named [{}]", name.toStdString());
        return false;
    }

    NODE_DATA data;
    data[ROLE_OBJID] = nodeid;
    data[ROLE_OBJNAME] = name;
    data[ROLE_COLLASPED] = false;
    data[ROLE_NODETYPE] = UiHelper::nodeType(name);

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
    bool ret = m_pModel->getDescriptor(name, desc);
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
        input.info.nodeid = id;
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
        output.info.nodeid = id;
        output.info.control = descOutput.info.control;
        output.info.ctrlProps = descOutput.info.ctrlProps;
        output.info.type = descOutput.info.type;
        output.info.name = descOutput.info.name;
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
        const rapidjson::Value& defaultValue
)
{
    const QString &subgName = m_currentGraph->name();
    QString outLinkPath;
    if (!outNode.isEmpty() && !outSock.isEmpty()) {
        outLinkPath = UiHelper::constructObjPath(subgName, outNode, "[node]/outputs/", outSock);
    }
    setInputSocket2(nodeCls, inNode, inSock, outLinkPath, "", defaultValue);
}

void ModelAcceptor::setInputSocket2(
                const QString& nodeCls,
                const QString& inNode,
                const QString& inSock,
                const QString& outLinkPath,
                const QString& sockProperty,
                const rapidjson::Value& defaultVal)
{
    if (!m_currentGraph)
        return;

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
        defaultValue = UiHelper::parseJsonByType(descInfo.type, defaultVal, m_currentGraph);
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
            m_subgLinks.append(fullLink);
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
                m_subgLinks.append(fullLink);
            }

            QModelIndex inNodeIdx = m_currentGraph->index(inNode);
            ZASSERT_EXIT(inNodeIdx.isValid());

            //the layout should be standard inputs desc by latest descriptors.

            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(inNodeIdx.data(ROLE_NODE_PARAMS));
            nodeParams->setAddParam(PARAM_INPUT, sockName, "string", "", CONTROL_NONE, QVariant(), prop);
        }
        else
        {
            zeno::log_warn("{}: no such input socket {}", nodeCls.toStdString(), inSock.toStdString());
        }
    }
}

void ModelAcceptor::setDictPanelProperty(bool bInput, const QString& ident, const QString& sockName, bool bCollasped)
{
    QModelIndex inNodeIdx = m_currentGraph->index(ident);
    ZASSERT_EXIT(inNodeIdx.isValid());

    QModelIndex sockIdx = m_currentGraph->nodeParamIndex(inNodeIdx, bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);
    ZASSERT_EXIT(sockIdx.isValid());

    DictKeyModel *keyModel = QVariantPtr<DictKeyModel>::asPtr(sockIdx.data(ROLE_VPARAM_LINK_MODEL));
    ZERROR_EXIT(keyModel);
    keyModel->setCollasped(bCollasped);
}

void ModelAcceptor::setControlAndProperties(const QString& nodeCls, const QString& inNode, const QString& inSock, PARAM_CONTROL control, const QVariant& ctrlProperties) 
{
    if (!m_currentGraph)
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
            const QString& link
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

void ModelAcceptor::setParamValue(const QString& id, const QString& nodeCls, const QString& name, const rapidjson::Value& value)
{
    if (!m_currentGraph)
        return;

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
