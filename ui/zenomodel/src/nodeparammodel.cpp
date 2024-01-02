#include "nodeparammodel.h"
#include "vparamitem.h"
#include "modelrole.h"
#include "globalcontrolmgr.h"
#include "uihelper.h"
#include "variantptr.h"
#include "globalcontrolmgr.h"
#include "dictkeymodel.h"
#include "iotags.h"
#include "dictkeymodel.h"
#include "common_def.h"
#include "subgraphmodel.h"


NodeParamModel::NodeParamModel(const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent)
    : ViewParamModel(nodeIdx, pModel, parent)
{
    initUI();
    connect(this, &NodeParamModel::modelAboutToBeReset, this, &NodeParamModel::onModelAboutToBeReset);
    connect(this, &NodeParamModel::rowsAboutToBeRemoved, this, &NodeParamModel::onRowsAboutToBeRemoved);
}

NodeParamModel::~NodeParamModel()
{
}

void NodeParamModel::onModelAboutToBeReset()
{
    clearParams();
}

void NodeParamModel::clearParams()
{
    if (auto inputs = getInputs()) {
        while (inputs->rowCount() > 0)
        {
            inputs->removeRows(0, 1);
        }
    }
    if (auto params = getParams()) {
        while (params->rowCount() > 0)
        {
            params->removeRows(0, 1);
        }
    }
    if (auto outputs = getOutputs()) {
        while (outputs->rowCount() > 0)
        {
            outputs->removeRows(0, 1);
        }
    }
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
    auto inputs = new VParamItem(VPARAM_GROUP, iotags::params::node_inputs);
    auto params = new VParamItem(VPARAM_GROUP, iotags::params::node_params);
    auto outputs = new VParamItem(VPARAM_GROUP, iotags::params::node_outputs);
    appendRow(inputs);
    appendRow(params);
    appendRow(outputs);
}

bool NodeParamModel::getInputSockets(INPUT_SOCKETS& inputs)
{
    auto pInputs = getInputs();
    ZASSERT_EXIT(pInputs, false);
    for (int r = 0; r < pInputs->rowCount(); r++)
    {
        VParamItem* param = static_cast<VParamItem*>(pInputs->child(r));
        const QString& name = param->m_name;

        INPUT_SOCKET inSocket;
        inSocket.info.defaultValue = param->data(ROLE_PARAM_VALUE);
        inSocket.info.nodeid = m_nodeIdx.data(ROLE_OBJID).toString();
        inSocket.info.name = param->m_name;
        inSocket.info.type = param->data(ROLE_PARAM_TYPE).toString();
        inSocket.info.sockProp = param->m_sockProp;
        inSocket.info.links = exportLinks(param->m_links);
        inSocket.info.control = param->m_ctrl;
        inSocket.info.ctrlProps = param->m_customData[ROLE_VPARAM_CTRL_PROPERTIES].toMap();
        inSocket.info.toolTip = param->m_customData[ROLE_VPARAM_TOOLTIP].toString();
        inSocket.info.netlabel = param->data(ROLE_PARAM_NETLABEL).toString();

        if (param->m_customData.find(ROLE_VPARAM_LINK_MODEL) != param->m_customData.end())
        {
            DictKeyModel* pModel = QVariantPtr<DictKeyModel>::asPtr(param->m_customData[ROLE_VPARAM_LINK_MODEL]);
            ZASSERT_EXIT(pModel, false);
            exportDictkeys(pModel, inSocket.info.dictpanel);
        }
        inputs.insert(name, inSocket);
    }
    return true;
}

bool NodeParamModel::getOutputSockets(OUTPUT_SOCKETS& outputs)
{
    auto pOutputs = getOutputs();
    ZASSERT_EXIT(pOutputs, false);
    for (int r = 0; r < pOutputs->rowCount(); r++)
    {
        VParamItem* param = static_cast<VParamItem*>(pOutputs->child(r));
        const QString& name = param->m_name;

        OUTPUT_SOCKET outSocket;
        outSocket.retIdx = param->index();

        outSocket.info.defaultValue = param->data(ROLE_PARAM_VALUE);
        outSocket.info.nodeid = m_nodeIdx.data(ROLE_OBJID).toString();
        outSocket.info.name = name;
        outSocket.info.type = param->data(ROLE_PARAM_TYPE).toString();
        outSocket.info.sockProp = param->m_sockProp;
        outSocket.info.links = exportLinks(param->m_links);
        outSocket.info.toolTip = param->m_customData[ROLE_VPARAM_TOOLTIP].toString();
        outSocket.info.netlabel = param->data(ROLE_PARAM_NETLABEL).toString();

        if (param->m_customData.find(ROLE_VPARAM_LINK_MODEL) != param->m_customData.end())
        {
            DictKeyModel* pModel = QVariantPtr<DictKeyModel>::asPtr(param->m_customData[ROLE_VPARAM_LINK_MODEL]);
            ZASSERT_EXIT(pModel, false);
            exportDictkeys(pModel, outSocket.info.dictpanel);
        }

        outputs.insert(name, outSocket);
    }
    return true;
}

bool NodeParamModel::getParams(PARAMS_INFO &params)
{
    auto pParams = getParams();
    if (!pParams)
        return false;

    for (int r = 0; r < pParams->rowCount(); r++)
    {
        VParamItem* param = static_cast<VParamItem*>(pParams->child(r));
        const QString& name = param->m_name;

        PARAM_INFO paramInfo;
        paramInfo.bEnableConnect = false;
        paramInfo.value = param->data(ROLE_PARAM_VALUE);
        paramInfo.typeDesc = param->data(ROLE_PARAM_TYPE).toString();
        paramInfo.name = name;
        paramInfo.control = param->m_ctrl;
        paramInfo.controlProps = param->m_customData[ROLE_VPARAM_CTRL_PROPERTIES];
        paramInfo.toolTip = param->m_customData[ROLE_VPARAM_TOOLTIP].toString();
        params.insert(name, paramInfo);
    }
    return true;
}

VParamItem* NodeParamModel::getInputs() const
{
    auto root = invisibleRootItem();
    for (int r = 0; r < root->rowCount(); r++)
    {
        auto pItem = static_cast<VParamItem*>(root->child(r));
        if (pItem->text() == iotags::params::node_inputs)
            return pItem;
    }
    return nullptr;
}

VParamItem* NodeParamModel::getParams() const
{
    auto root = invisibleRootItem();
    for (int r = 0; r < root->rowCount(); r++)
    {
        auto pItem = static_cast<VParamItem*>(root->child(r));
        if (pItem->text() == iotags::params::node_params)
            return pItem;
    }
    return nullptr;
}

VParamItem* NodeParamModel::getOutputs() const
{
    auto root = invisibleRootItem();
    for (int r = 0; r < root->rowCount(); r++)
    {
        auto pItem = static_cast<VParamItem*>(root->child(r));
        if (pItem->text() == iotags::params::node_outputs)
            return pItem;
    }
    return nullptr;
}

VParamItem* NodeParamModel::getLegacyInputs() const
{
    auto root = invisibleRootItem();
    for (int r = 0; r < root->rowCount(); r++)
    {
        auto pItem = static_cast<VParamItem*>(root->child(r));
        if (pItem->text() == iotags::params::legacy_inputs)
            return pItem;
    }
    return nullptr;
}

VParamItem* NodeParamModel::getLegacyParams() const
{
    auto root = invisibleRootItem();
    for (int r = 0; r < root->rowCount(); r++)
    {
        auto pItem = static_cast<VParamItem*>(root->child(r));
        if (pItem->text() == iotags::params::legacy_params)
            return pItem;
    }
    return nullptr;
}

VParamItem* NodeParamModel::getLegacyOutputs() const
{
    auto root = invisibleRootItem();
    for (int r = 0; r < root->rowCount(); r++)
    {
        auto pItem = static_cast<VParamItem*>(root->child(r));
        if (pItem->text() == iotags::params::legacy_outputs)
            return pItem;
    }
    return nullptr;
}

QModelIndexList NodeParamModel::getInputIndice() const
{
    QModelIndexList lst;
    auto pInputs = getInputs();
    ZASSERT_EXIT(pInputs, lst);

    for (int i = 0; i < pInputs->rowCount(); i++) {
        lst.append(pInputs->child(i)->index());
    }
    return lst;
}

QModelIndexList NodeParamModel::getParamIndice() const
{
    QModelIndexList lst;
    auto pParams = getParams();
    if (!pParams)
        return lst;

    for (int i = 0; i < pParams->rowCount(); i++) {
        lst.append(pParams->child(i)->index());
    }
    return lst;
}

QModelIndexList NodeParamModel::getOutputIndice() const
{
    QModelIndexList lst;
    auto pOutputs = getOutputs();
    ZASSERT_EXIT(pOutputs, lst);

    for (int i = 0; i < pOutputs->rowCount(); i++) {
        lst.append(pOutputs->child(i)->index());
    }
    return lst;
}

void NodeParamModel::setInputSockets(const INPUT_SOCKETS& inputs)
{
    for (INPUT_SOCKET inSocket : inputs)
    {
        setAddParam(PARAM_INPUT, 
                    inSocket.info.name,
                    inSocket.info.type,
                    inSocket.info.defaultValue,
                    inSocket.info.control,
                    inSocket.info.ctrlProps,
                    (SOCKET_PROPERTY)inSocket.info.sockProp,
                    inSocket.info.dictpanel,
                    inSocket.info.toolTip,
                    inSocket.info.netlabel
            );
    }
}

void NodeParamModel::setParams(const PARAMS_INFO& params)
{
    for (PARAM_INFO paramInfo : params)
    {
        setAddParam(PARAM_PARAM,
            paramInfo.name,
            paramInfo.typeDesc,
            paramInfo.value,
            paramInfo.control,
            paramInfo.controlProps,
            SOCKPROP_UNKNOWN,
            DICTPANEL_INFO(),
            paramInfo.toolTip
        );
    }
}

void NodeParamModel::setOutputSockets(const OUTPUT_SOCKETS& outputs)
{
    for (OUTPUT_SOCKET outSocket : outputs)
    {
        setAddParam(PARAM_OUTPUT,
            outSocket.info.name,
            outSocket.info.type,
            outSocket.info.defaultValue,
            outSocket.info.control,
            outSocket.info.ctrlProps,
            (SOCKET_PROPERTY)outSocket.info.sockProp,
            outSocket.info.dictpanel,
            outSocket.info.toolTip,
            outSocket.info.netlabel);
    }
}

QList<EDGE_INFO> NodeParamModel::exportLinks(const PARAM_LINKS& links)
{
    QList<EDGE_INFO> linkInfos;
    for (auto linkIdx : links)
    {
        EDGE_INFO link = exportLink(linkIdx);
        linkInfos.append(link);
    }
    return linkInfos;
}

EDGE_INFO NodeParamModel::exportLink(const QModelIndex& linkIdx)
{
    EDGE_INFO link;

    QModelIndex outSock = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
    QModelIndex inSock = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    ZASSERT_EXIT(outSock.isValid() && inSock.isValid(), link);

    link.outSockPath = outSock.data(ROLE_OBJPATH).toString();
    link.inSockPath = inSock.data(ROLE_OBJPATH).toString();
    return link;
}

void NodeParamModel::removeParam(PARAM_CLASS cls, const QString& name)
{
    if (PARAM_INPUT == cls)
    {
        if (auto pInputs = getInputs())
        {
            for (int i = 0; i < pInputs->rowCount(); i++)
            {
                VParamItem* pChild = static_cast<VParamItem*>(pInputs->child(i));
                if (pChild->m_name == name)
                {
                    pInputs->removeRow(i);
                    return;
                }
            }
        }
    }
    if (PARAM_PARAM == cls)
    {
        if (auto pParams = getParams())
        {
            for (int i = 0; i < pParams->rowCount(); i++)
            {
                VParamItem* pChild = static_cast<VParamItem*>(pParams->child(i));
                if (pChild->m_name == name)
                {
                    pParams->removeRow(i);
                    return;
                }
            }
        }
    }
    if (PARAM_OUTPUT == cls)
    {
        if (auto pOutputs = getOutputs())
        {
            for (int i = 0; i < pOutputs->rowCount(); i++)
            {
                VParamItem* pChild = static_cast<VParamItem*>(pOutputs->child(i));
                if (pChild->m_name == name)
                {
                    pOutputs->removeRow(i);
                    return;
                }
            }
        }
    }
}

void NodeParamModel::setAddParam(
                PARAM_CLASS cls,
                const QString& name,
                const QString& type,
                const QVariant& deflValue,
                PARAM_CONTROL ctrl,
                QVariant ctrlProps,
                SOCKET_PROPERTY prop,
                DICTPANEL_INFO dictPanel,
                const QString& toolTip,
                const QString& netLabel)
{
    VParamItem *pItem = nullptr;
    const QString& nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();

    VParamItem* pGroup = nullptr;
    switch (cls) {
    case PARAM_INPUT: pGroup = getInputs(); break;
    case PARAM_PARAM: pGroup = getParams(); break;
    case PARAM_OUTPUT: pGroup = getOutputs(); break;
    case PARAM_LEGACY_INPUT:
    {
        pGroup = getLegacyInputs();
        if (!pGroup) {
            pGroup = new VParamItem(VPARAM_GROUP, iotags::params::legacy_inputs);
            appendRow(pGroup);
        }
        break;
    }
    case PARAM_LEGACY_PARAM:
    {
        pGroup = getLegacyParams();
        if (!pGroup) {
            pGroup = new VParamItem(VPARAM_GROUP, iotags::params::legacy_params);
            appendRow(pGroup);
        }
        break;
    }
    case PARAM_LEGACY_OUTPUT:
    {
        pGroup = getLegacyOutputs();
        if (!pGroup) {
            pGroup = new VParamItem(VPARAM_GROUP, iotags::params::legacy_outputs);
            appendRow(pGroup);
        }
        break;
    }
    }

    ZASSERT_EXIT(pGroup);

    if (!(pItem = pGroup->getItem(name)))
    {
        pItem = new VParamItem(VPARAM_PARAM, name);
        ZASSERT_EXIT(pItem);
        pItem->setData(ctrlProps, ROLE_VPARAM_CTRL_PROPERTIES);
        pItem->setData(toolTip, ROLE_VPARAM_TOOLTIP);
        pItem->setData(name, ROLE_PARAM_NAME);
        pItem->setData(deflValue, ROLE_PARAM_VALUE);
        pItem->setData(type, ROLE_PARAM_TYPE);
        pItem->m_sockProp = prop;
        pItem->setData(ctrl, ROLE_PARAM_CTRL);
        pGroup->appendRow(pItem);
        if (!netLabel.isEmpty())
        {
            const QModelIndex& subgIdx = m_nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
            m_pGraphsModel->addNetLabel(subgIdx, pItem->index(), netLabel);
        }
        if (PARAM_PARAM != cls)
            initDictSocket(pItem, dictPanel);
    }
    else
    {
        pItem->setData(deflValue, ROLE_PARAM_VALUE);
        pItem->m_name = name;
        pItem->setData(type, ROLE_PARAM_TYPE);      //only allow to change type on IO processing, especially for SubInput.
        pItem->m_sockProp = prop;
        pItem->setData(ctrl, ROLE_PARAM_CTRL);
        pItem->setData(ctrlProps, ROLE_VPARAM_CTRL_PROPERTIES);
        pItem->setData(toolTip, ROLE_VPARAM_TOOLTIP);
        if (!netLabel.isEmpty())
        {
            const QModelIndex& subgIdx = m_nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
            m_pGraphsModel->addNetLabel(subgIdx, pItem->index(), netLabel);
        }

        if (PARAM_PARAM != cls && 
            pItem->m_customData.find(ROLE_VPARAM_LINK_MODEL) != pItem->m_customData.end())
        {
            DictKeyModel* pDictModel = QVariantPtr<DictKeyModel>::asPtr(pItem->m_customData[ROLE_VPARAM_LINK_MODEL]);
            if (pDictModel)
            {
                for (int r = 0; r < dictPanel.keys.size(); r++) {
                    const DICTKEY_INFO &keyInfo = dictPanel.keys[r];
                    pDictModel->insertRow(r);
                    QModelIndex newIdx = pDictModel->index(r, 0);
                    pDictModel->setData(newIdx, keyInfo.key, ROLE_PARAM_NAME);
                    if (!keyInfo.netLabel.isEmpty())
                    {
                        const QModelIndex& subgIdx = m_nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
                        m_pGraphsModel->addNetLabel(subgIdx, newIdx, keyInfo.netLabel);
                    }
                }
            }
        }
    }
}

QVariant NodeParamModel::getValue(PARAM_CLASS cls, const QString& name) const
{
    VParamItem *pItem = nullptr;
    if (PARAM_INPUT == cls)
    {
        auto pInputs = getInputs();
        ZASSERT_EXIT(pInputs, QVariant());
        pItem = pInputs->getItem(name);
    }
    else if (PARAM_PARAM == cls)
    {
        auto pParams = getParams();
        if (pParams) {
            pItem = pParams->getItem(name);
        }
    }
    else if (PARAM_OUTPUT == cls)
    {
        auto pOutputs = getOutputs();
        ZASSERT_EXIT(pOutputs, QVariant());
        pItem = pOutputs->getItem(name);
    }

    if (!pItem)
        return QVariant();

    return pItem->data(ROLE_PARAM_VALUE);
}

QModelIndex NodeParamModel::getParam(PARAM_CLASS cls, const QString& name) const
{
    //todo: inner dict key.
    if (PARAM_INPUT == cls)
    {
        auto pInputs = getInputs();
        ZASSERT_EXIT(pInputs, QModelIndex());
        if (VParamItem* pItem = pInputs->getItem(name))
        {
            return pItem->index();
        }
    }
    else if (PARAM_PARAM == cls)
    {
        auto pParams = getParams();
        if (pParams)
        {
            if (VParamItem* pItem = pParams->getItem(name))
            {
                return pItem->index();
            }
        }
    }
    else if (PARAM_OUTPUT == cls)
    {
        auto pOutputs = getOutputs();
        ZASSERT_EXIT(pOutputs, QModelIndex());
        if (VParamItem* pItem = pOutputs->getItem(name))
        {
            return pItem->index();
        }
    }
    else if (PARAM_LEGACY_INPUT == cls)
    {
        auto pInputs = getLegacyInputs();
        if (pInputs)
        {
            if (VParamItem* pItem = pInputs->getItem(name))
                return pItem->index();
        }
    }
    else if (PARAM_LEGACY_PARAM == cls)
    {
        auto pParams = getLegacyParams();
        if (pParams)
        {
            if (VParamItem* pItem = pParams->getItem(name))
                return pItem->index();
        }
    }
    else if (PARAM_LEGACY_OUTPUT == cls)
    {
        auto pOutputs = getLegacyOutputs();
        if (pOutputs)
        {
            if (VParamItem* pItem = pOutputs->getItem(name))
                return pItem->index();
        }
    }
    return QModelIndex();
}

QVariant NodeParamModel::data(const QModelIndex& index, int role) const
{
    VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(index));
    if (!pItem)
        return QVariant();

    switch (role)
    {
    case ROLE_OBJPATH:
    {
        QString path;
        QStandardItem* pItem = itemFromIndex(index);
        do
        {
            path = pItem->data(ROLE_VPARAM_NAME).toString() + path;
            path = "/" + path;
            pItem = pItem->parent();
        } while (pItem);
        path = "[node]" + path;
        path = m_nodeIdx.data(ROLE_OBJPATH).toString() + cPathSeperator + path;
        return path;
    }
    case ROLE_PARAM_CLASS:
    {
        if (pItem->vType != VPARAM_PARAM)
            return QVariant();
        VParamItem* parentItem = static_cast<VParamItem*>(pItem->parent());
        if (iotags::params::node_inputs == parentItem->m_name)
            return PARAM_INPUT;
        else if (iotags::params::node_outputs == parentItem->m_name)
            return PARAM_OUTPUT;
        else if (iotags::params::node_params == parentItem->m_name)
            return PARAM_PARAM;
        else if (iotags::params::legacy_inputs == parentItem->m_name)
            return PARAM_LEGACY_INPUT;
        else if (iotags::params::legacy_params == parentItem->m_name)
            return PARAM_LEGACY_PARAM;
        else if (iotags::params::legacy_outputs == parentItem->m_name)
            return PARAM_LEGACY_OUTPUT;
        return PARAM_UNKNOWN;
    }
    case ROLE_VPARAM_LINK_MODEL:
    case ROLE_VPARAM_CTRL_PROPERTIES:
    {
        if (pItem->m_customData.find(role) != pItem->m_customData.end())
        {
            return pItem->m_customData[role];
        }
        return QVariant();
    }
    default:
        return ViewParamModel::data(index, role);
    }

}

bool NodeParamModel::isEditable(const QModelIndex& current)
{
    bool bCoreParam = current.data(ROLE_VPARAM_IS_COREPARAM).toBool();
    if (bCoreParam)
        return false;
    int type = current.data(ROLE_VPARAM_TYPE).toInt();
    if (type == VPARAM_GROUP)
    {
        return false;
    }
    else if (type == VPARAM_PARAM)
    {
        if (!m_pGraphsModel->IsSubGraphNode(m_nodeIdx))
            return isEditable(current.parent());
    }
    return true;
}

QModelIndex NodeParamModel::indexFromPath(const QString& path)
{
    QStringList lst = path.split("/", QtSkipEmptyParts);
    if (lst.size() < 2)
        return QModelIndex();

    const QString& group = lst[0];
    const QString& name = lst[1];
    QString subkey = lst.size() > 2 ? lst[2] : "";

    if (group == iotags::params::node_inputs)
    {
        auto pInputs = getInputs();
        ZASSERT_EXIT(pInputs, QModelIndex());
        if (VParamItem* pItem = pInputs->getItem(name))
        {
            if (!subkey.isEmpty())
            {
                if (pItem->m_customData.find(ROLE_VPARAM_LINK_MODEL) != pItem->m_customData.end())
                {
                    DictKeyModel* keyModel = QVariantPtr<DictKeyModel>::asPtr(pItem->m_customData[ROLE_VPARAM_LINK_MODEL]);
                    ZASSERT_EXIT(keyModel, QModelIndex());
                    return keyModel->index(subkey);
                }
            }
            return pItem->index();
        }
    }
    else if (group == iotags::params::legacy_inputs)
    {
        auto plegacyInputs = getLegacyInputs();
        if (plegacyInputs)
        {
            if (VParamItem* pItem = plegacyInputs->getItem(name))
            {
                return pItem->index();
            }
        }
    }
    else if (group == iotags::params::node_params)
    {
        auto pParams = getParams();
        if (!pParams)
            return QModelIndex();
        if (VParamItem* pItem = pParams->getItem(name))
        {
            return pItem->index();
        }
    }
    else if (group == iotags::params::legacy_params)
    {
        auto plegacyParams = getLegacyParams();
        if (plegacyParams) {
            if (VParamItem* pItem = plegacyParams->getItem(name))
                return pItem->index();
        }
    }
    else if (group == iotags::params::node_outputs)
    {
        auto pOutputs = getOutputs();
        ZASSERT_EXIT(pOutputs, QModelIndex());
        if (VParamItem* pItem = pOutputs->getItem(name))
        {
            if (!subkey.isEmpty())
            {
                if (pItem->m_customData.find(ROLE_VPARAM_LINK_MODEL) != pItem->m_customData.end())
                {
                    DictKeyModel* keyModel = QVariantPtr<DictKeyModel>::asPtr(pItem->m_customData[ROLE_VPARAM_LINK_MODEL]);
                    ZASSERT_EXIT(keyModel, QModelIndex());
                    return keyModel->index(subkey);
                }
            }
            return pItem->index();
        }
    }
    else if (group == iotags::params::legacy_outputs)
    {
        auto plegacyOutputs = getLegacyOutputs();
        if (plegacyOutputs) {
            if (VParamItem* pItem = plegacyOutputs->getItem(name))
                return pItem->index();
        }
    }
    return QModelIndex();
}

QStringList NodeParamModel::sockNames(PARAM_CLASS cls) const
{
    QStringList names;
    if (cls == PARAM_INPUT)
    {
        auto pInputs = getInputs();
        ZASSERT_EXIT(pInputs, names);
        for (int r = 0; r < pInputs->rowCount(); r++)
        {
            VParamItem* pItem = static_cast<VParamItem*>(pInputs->child(r));
            names.append(pItem->m_name);
        }
    }
    else if (cls == PARAM_PARAM)
    {
        auto pParams = getParams();
        if (!pParams)
            return names;

        for (int r = 0; r < pParams->rowCount(); r++)
        {
            VParamItem* pItem = static_cast<VParamItem*>(pParams->child(r));
            names.append(pItem->m_name);
        }
    }
    else if (cls == PARAM_OUTPUT)
    {
        auto pOutputs = getOutputs();
        ZASSERT_EXIT(pOutputs, names);
        for (int r = 0; r < pOutputs->rowCount(); r++)
        {
            VParamItem* pItem = static_cast<VParamItem*>(pOutputs->child(r));
            names.append(pItem->m_name);
        }
    }
    return names;
}

void NodeParamModel::clone(ViewParamModel* pModel)
{
    auto pInputs = getInputs();
    auto pParams = getParams();
    auto pOutputs = getOutputs();

    //only add params.
    NodeParamModel* pOtherModel = qobject_cast<NodeParamModel*>(pModel);
    auto pInputs_ = pOtherModel->getInputs();
    auto pParams_ = pOtherModel->getParams();
    auto pOutputs_ = pOtherModel->getOutputs();

    ZASSERT_EXIT(pInputs && pInputs_ && pOutputs && pOutputs_);

    for (int i = 0; i < pInputs_->rowCount(); i++)
    {
        QStandardItem* item = pInputs_->child(i)->clone();
        pInputs->appendRow(item);
    }

    if (pParams_) {
        for (int i = 0; i < pParams_->rowCount(); i++)
        {
            QStandardItem* item = pParams_->child(i)->clone();
            if (pParams)
                pParams->appendRow(item);
        }
    }

    for (int i = 0; i < pOutputs_->rowCount(); i++)
    {
        QStandardItem* item = pOutputs_->child(i)->clone();
        pOutputs->appendRow(item);
    }

    pInputs->m_uuid = pInputs_->m_uuid;
    if (pParams && pParams_)
        pParams->m_uuid = pParams_->m_uuid;
    pOutputs->m_uuid = pOutputs_->m_uuid;
}

bool NodeParamModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    switch (role)
    {
        case ROLE_PARAM_NAME:
        {
            VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            const QString& oldName = pItem->m_name;
            QString newName = value.toString();
            if (oldName == newName)
                return false;

            checkExtractDict(newName);
            pItem->setData(newName, role);
            markNodeChanged();
            break;
        }
        case ROLE_PARAM_TYPE:
        {
            VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            if (pItem->data(ROLE_PARAM_TYPE) == value)
                return false;

            pItem->setData(value, role);
            markNodeChanged();
            break;
        }
        case ROLE_PARAM_VALUE:
        {
            VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            QVariant oldValue = pItem->data(ROLE_PARAM_VALUE);
            if (oldValue == value && oldValue.type() == value.type())
                return false;

            pItem->setData(value, role);
            onSubIOEdited(oldValue, pItem);
            markNodeChanged();
            break;
        }
        case ROLE_ADDLINK:
        case ROLE_REMOVELINK:
        {
            VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            if (pItem->vType != VPARAM_PARAM)
                return false;

            pItem->setData(value, role);
            if (role == ROLE_ADDLINK)
            {
                onLinkAdded(pItem);
            }
            markNodeChanged();
            break;
        }
        case ROLE_PARAM_CTRL: {
            VParamItem *pItem = static_cast<VParamItem *>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            QVariant oldValue = pItem->m_ctrl;
            if (oldValue == value)
                return false;

            pItem->setData(value, role);
            onSubIOEdited(oldValue, pItem);
            break;
        }
        case ROLE_VPARAM_CTRL_PROPERTIES: {
            VParamItem *pItem = static_cast<VParamItem *>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            QVariant oldValue = pItem->m_customData[role];
            if (oldValue == value)
                return false;

            pItem->setData(value, role);
            onSubIOEdited(oldValue, pItem);
            break;
        }
        case ROLE_PARAM_NETLABEL:
        {
            VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(index));
            ZERROR_EXIT(pItem, false);
            QVariant oldValue = pItem->m_customData[role];
            if (oldValue == value)
                return false;

            pItem->setData(value, role);
            break;
        }
    default:
        return ViewParamModel::setData(index, value, role);
    }
    return false;
}

void NodeParamModel::clearLinks(VParamItem* pItem)
{
    PARAM_LINKS links = pItem->m_links;
    for (const QPersistentModelIndex& linkIdx : links)
    {
        m_pGraphsModel->removeLink(linkIdx, true);
    }
    pItem->m_links.clear();
}

void NodeParamModel::initDictSocket(VParamItem* pItem, const DICTPANEL_INFO& dictpanel)
{
    if (!pItem || pItem->vType != VPARAM_PARAM)
        return;

    const QString& nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();
    NODE_DESC desc;
    m_pGraphsModel->getDescriptor(nodeCls, desc);

    const QString& paramType = pItem->data(ROLE_PARAM_TYPE).toString();

    if (paramType == "dict" || paramType == "DictObject" || paramType == "DictObject:NumericObject")
    {
        pItem->setData("dict", ROLE_PARAM_TYPE);    //pay attention not to export to outside, only as a ui keyword.
        if (!desc.categories.contains("dict") || nodeCls == "MultiMakeDict")
            pItem->m_sockProp = SOCKPROP_DICTLIST_PANEL;
    }
    else if (paramType == "list")
    {
        PARAM_CLASS cls = pItem->getParamClass();
        if ((!desc.categories.contains("list") || nodeCls == "MultiMakeList") && cls == PARAM_INPUT)
            pItem->m_sockProp = SOCKPROP_DICTLIST_PANEL;
    } 
    else if (paramType == "group-line") 
    {
        if (!desc.categories.contains("group-line"))
            pItem->m_sockProp = SOCKPROP_GROUP_LINE;
    }

    //not type desc on list output socket, add it here.
    if (pItem->m_name == "list" && paramType.isEmpty())
    {
        pItem->setData("list", ROLE_PARAM_TYPE);
        PARAM_CLASS cls = pItem->getParamClass();
        if (cls == PARAM_INPUT && !desc.categories.contains("list"))
            pItem->m_sockProp = SOCKPROP_DICTLIST_PANEL;
    }

    if (pItem->m_sockProp == SOCKPROP_DICTLIST_PANEL)
    {
        DictKeyModel* pDictModel = new DictKeyModel(m_pGraphsModel, pItem->index(), this);
        for (int r = 0; r < dictpanel.keys.size(); r++)
        {
            const DICTKEY_INFO& keyInfo = dictpanel.keys[r];
            pDictModel->insertRow(r);
            QModelIndex newIdx = pDictModel->index(r, 0);
            pDictModel->setData(newIdx, keyInfo.key, ROLE_PARAM_NAME);
            if (!keyInfo.netLabel.isEmpty())
            {
                const QModelIndex& subgIdx = m_nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
                m_pGraphsModel->addNetLabel(subgIdx, newIdx, keyInfo.netLabel);
            }
        }
        pItem->m_customData[ROLE_VPARAM_LINK_MODEL] = QVariantPtr<DictKeyModel>::asVariant(pDictModel);
    }
}

void NodeParamModel::exportDictkeys(DictKeyModel* pModel, DICTPANEL_INFO& panel)
{
    if (!pModel)
        return;

    panel.bCollasped = pModel->isCollasped();

    int rowCnt = pModel->rowCount();
    QStringList keyNames;
    for (int i = 0; i < rowCnt; i++)
    {
        const QModelIndex &keyIdx = pModel->index(i, 0);
        QString key = keyIdx.data().toString();

        DICTKEY_INFO keyInfo;
        keyInfo.key = key;
        keyInfo.netLabel = keyIdx.data(ROLE_PARAM_NETLABEL).toString();

        PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
        for (auto linkIdx : links)
        {
            if (linkIdx.isValid())
            {
                QModelIndex outsock = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
                QModelIndex insock = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
                ZASSERT_EXIT(insock.isValid() && outsock.isValid());

                EDGE_INFO link = exportLink(linkIdx);
                if (link.isValid())
                    keyInfo.links.append(link);
            }
        }

        panel.keys.append(keyInfo);
        keyNames.push_back(key);
    }
}

void NodeParamModel::checkExtractDict(QString &name)
{
    QString nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();
    QStringList lst = name.split(",");
    if (nodeCls == "ExtractDict" && lst.size() > 1) 
    {
        for (int i = 1; i < lst.size(); i++)
        {
            setAddParam(PARAM_OUTPUT, lst.at(i).simplified(), "", QVariant(), CONTROL_NONE, QVariant(), SOCKPROP_EDITABLE);
        }
        name = lst.first();
    }
}

void NodeParamModel::markNodeChanged()
{
    m_pGraphsModel->markNodeDataChanged(m_nodeIdx);
}

void NodeParamModel::onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    VParamItem* parentItem = static_cast<VParamItem*>(itemFromIndex(parent));
    if (!parentItem || parentItem->vType != VPARAM_GROUP)
        return;

    if (first < 0 || first >= parentItem->rowCount())
        return;

    //todo: begin macro

    VParamItem* pItem = static_cast<VParamItem*>(parentItem->child(first));
    clearLinks(pItem);

    //clear subkeys.
    if (pItem->m_customData.find(ROLE_VPARAM_LINK_MODEL) != pItem->m_customData.end())
    {
        DictKeyModel* keyModel = QVariantPtr<DictKeyModel>::asPtr(pItem->m_customData[ROLE_VPARAM_LINK_MODEL]);
        keyModel->clearAll();
    }
}

bool NodeParamModel::removeRows(int row, int count, const QModelIndex& parent)
{
    VParamItem* parentItem = static_cast<VParamItem*>(itemFromIndex(parent));
    if (!parentItem || parentItem->vType != VPARAM_GROUP)
        return false;

    if (row < 0 || row >= parentItem->rowCount())
        return false;

    VParamItem* pItem = static_cast<VParamItem*>(parentItem->child(row));
    clearLinks(pItem);

    bool ret = ViewParamModel::removeRows(row, count, parent);
    m_pGraphsModel->markDirty();
    return ret;
}

bool NodeParamModel::isTempModel()
{
    //temp model on edit param dialog, no actual operation to the graph.
    if (qobject_cast<SubGraphModel*>(parent()))
        return false;
    else
        return true;
}

void NodeParamModel::onSubIOEdited(const QVariant& oldValue, const VParamItem* pItem)
{
    if (m_pGraphsModel->IsIOProcessing() || isTempModel())
        return;

    const QString& nodeName = m_nodeIdx.data(ROLE_OBJNAME).toString();
    if (nodeName == "SubInput" || nodeName == "SubOutput")
    {
        bool bInput = nodeName == "SubInput";
        QModelIndex subgIdx = m_nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
        ZASSERT_EXIT(subgIdx.isValid());
        const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();

        auto params = getParams();
        ZASSERT_EXIT(params);

        VParamItem* deflItem = params->getItem("defl");
        VParamItem* nameItem = params->getItem("name");
        VParamItem* typeItem = params->getItem("type");

        ZASSERT_EXIT(deflItem && nameItem && typeItem);
        const QString &sockName = nameItem->data(ROLE_PARAM_VALUE).toString();

        if (pItem->m_name == "type")
        {
            const QString& newType = pItem->data(ROLE_PARAM_VALUE).toString();
            PARAM_CONTROL newCtrl = UiHelper::getControlByType(newType);
            const QVariant& newValue = UiHelper::initDefaultValue(newType);

            const QModelIndex& idx_defl = deflItem->index();
            setData(idx_defl, newType, ROLE_PARAM_TYPE);
            setData(idx_defl, newCtrl, ROLE_PARAM_CTRL);
            setData(idx_defl, newValue, ROLE_PARAM_VALUE);

            //update desc.
            NODE_DESC desc;
            bool ret = m_pGraphsModel->getDescriptor(subgName, desc);
            ZASSERT_EXIT(ret);
            if (bInput)
            {
                ZASSERT_EXIT(desc.inputs.find(sockName) != desc.inputs.end());
                desc.inputs[sockName].info.type = newType;
                desc.inputs[sockName].info.control = newCtrl;
            }
            else
            {
                ZASSERT_EXIT(desc.outputs.find(sockName) != desc.outputs.end());
                desc.outputs[sockName].info.type = newType;
            }
            m_pGraphsModel->updateSubgDesc(subgName, desc);

            //update type of port. output need this?
            if (nodeName == "SubInput") {
                auto outputs = getOutputs();
                ZASSERT_EXIT(outputs);
                VParamItem *portItem = outputs->getItem("port");
                if (portItem) {
                    portItem->setData(newType, ROLE_PARAM_TYPE);
                }
            }

            //update to every subgraph node.
            QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subgName);
            for (auto idx : subgNodes)
            {
                // update socket for current subgraph node.
                NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
                QModelIndex paramIdx = nodeParams->getParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);
                nodeParams->setData(paramIdx, newType, ROLE_PARAM_TYPE);
                nodeParams->setData(paramIdx, newCtrl, ROLE_PARAM_CTRL);
                nodeParams->setData(paramIdx, newValue, ROLE_PARAM_VALUE);
            }
        }
        else if (pItem->m_name == "name")
        {
            //1.update desc info for the subgraph node.
            const QString& newName = sockName;
            const QString& oldName = oldValue.toString();

            NODE_DESC desc;
            bool ret = m_pGraphsModel->getDescriptor(subgName, desc);
            ZASSERT_EXIT(ret && newName != oldName);
            if (bInput)
            {
                desc.inputs[newName] = desc.inputs[oldName];
                desc.inputs[newName].info.name = newName;
                desc.inputs.remove(oldName);
            }
            else
            {
                desc.outputs[newName] = desc.outputs[oldName];
                desc.outputs[newName].info.name = newName;
                desc.outputs.remove(oldName);
            }
            m_pGraphsModel->updateSubgDesc(subgName, desc);

            //2.update all sockets for all subgraph node.
            QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subgName);
            for (auto idx : subgNodes)
            {
                // update socket for current subgraph node.
                NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
                QModelIndex paramIdx = nodeParams->getParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, oldName);
                nodeParams->setData(paramIdx, newName, ROLE_PARAM_NAME);
            }
        }
        else if (pItem->m_name == "defl")
        {
            const QVariant& deflVal = pItem->data(ROLE_PARAM_VALUE);
            NODE_DESC desc;
            bool ret = m_pGraphsModel->getDescriptor(subgName, desc);
            ZASSERT_EXIT(ret);
            bool isUpdate = false;
            if (bInput)
            {
                ZASSERT_EXIT(desc.inputs.find(sockName) != desc.inputs.end());
                desc.inputs[sockName].info.defaultValue = deflVal;
                QVariantMap ctrlProp = pItem->m_customData[ROLE_VPARAM_CTRL_PROPERTIES].toMap();
                if (desc.inputs[sockName].info.control != pItem->m_ctrl) 
                {
                    desc.inputs[sockName].info.control = pItem->m_ctrl;
                    isUpdate = true;
                }
                if (desc.inputs[sockName].info.ctrlProps != ctrlProp)
                {
                    desc.inputs[sockName].info.ctrlProps = ctrlProp;
                    isUpdate = true;
				}
            }
            else
            {
                ZASSERT_EXIT(desc.outputs.find(sockName) != desc.outputs.end());
                desc.outputs[sockName].info.defaultValue = deflVal;
            }
            m_pGraphsModel->updateSubgDesc(subgName, desc);
            //no need to update all subgraph node because it causes disturbance.
            //update all subgraph when ctrl properties changed
            if (isUpdate) 
			{
                QModelIndexList subgNodes = m_pGraphsModel->findSubgraphNode(subgName);
                for (auto idx : subgNodes) {
                    // update socket for current subgraph node.
                    NodeParamModel *nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
                    QModelIndex paramIdx = nodeParams->getParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, sockName);
                    QVariant ctrlProp = pItem->m_customData[ROLE_VPARAM_CTRL_PROPERTIES];
                    nodeParams->setData(paramIdx, pItem->data(ROLE_PARAM_TYPE), ROLE_PARAM_TYPE);
                    nodeParams->setData(paramIdx, pItem->m_ctrl, ROLE_PARAM_CTRL);
                    nodeParams->setData(paramIdx, ctrlProp, ROLE_VPARAM_CTRL_PROPERTIES);
                    QVariantMap props = ctrlProp.toMap();
                    if (props.find("items") != props.end()) {
                        QStringList items = props["items"].toStringList();
                        if (!items.contains(paramIdx.data(ROLE_PARAM_VALUE).toString())) {
                            nodeParams->setData(paramIdx, deflVal, ROLE_PARAM_VALUE);
                        }
                    }
                }
			}
        }
    }
}

void NodeParamModel::onLinkAdded(VParamItem* pItem)
{
    if (isTempModel())
        return;

    if (pItem->getParamClass() == PARAM_INPUT &&
        !pItem->data(ROLE_PARAM_NETLABEL).toString().isEmpty())
    {
        //todo: remove label from subgraphmodel::m_labels.
        pItem->setData("", ROLE_PARAM_NETLABEL);
    }

    //dynamic socket from MakeList/Dict and ExtractDict
    QString nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();
    QStringList lst;
    if (nodeCls == "ExtractDict") {
        lst = sockNames(PARAM_OUTPUT);
    } else {
        lst = sockNames(PARAM_INPUT);
    }
    int maxObjId = UiHelper::getMaxObjId(lst);
    if (maxObjId == -1)
        maxObjId = 0;

    QString maxObjSock = QString("obj%1").arg(maxObjId);
    QString lastKey = lst.last();
    if ((nodeCls == "MakeList" || nodeCls == "MakeDict") && pItem->m_name == lastKey)
    {
        const QString &newObjName = QString("obj%1").arg(maxObjId + 1);
        SOCKET_PROPERTY prop = nodeCls == "MakeDict" ? SOCKPROP_EDITABLE : SOCKPROP_NORMAL;
        setAddParam(PARAM_INPUT, newObjName, "", QVariant(), CONTROL_NONE, QVariant(), prop);
    }
    else if (nodeCls == "ExtractDict" && pItem->m_name == lastKey)
    {
        const QString &newObjName = QString("obj%1").arg(maxObjId + 1);
        setAddParam(PARAM_OUTPUT, newObjName, "", QVariant(), CONTROL_NONE, QVariant(), SOCKPROP_EDITABLE);
    }
}
