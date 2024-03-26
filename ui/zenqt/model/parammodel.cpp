#include "parammodel.h"
#include "zassert.h"
#include "util/uihelper.h"
#include <zeno/core/data.h>
#include <zeno/core/IParam.h>
#include "model/LinkModel.h"
#include "model/GraphModel.h"
#include "variantptr.h"


ParamsModel::ParamsModel(std::shared_ptr<zeno::INode> spNode, QObject* parent)
    : QAbstractListModel(parent)
    , m_wpNode(spNode)
    , m_customParamsM(nullptr)
{
    initParamItems();

    //TODO: register callback for core param adding/removing, for the functionally of custom param panel.
    cbUpdateParam = spNode->register_update_param(
        [this](const std::string& name, zeno::zvariant old_value, zeno::zvariant new_value) {
            for (int i = 0; i < m_items.size(); i++) {
                if (m_items[i].name.toStdString() == name) {
                    QVariant newValue = UiHelper::zvarToQVar(new_value);
                    m_items[i].value = newValue; //update cache
                    QModelIndex idx = createIndex(i, 0);
                    emit dataChanged(idx, idx, { ROLE_PARAM_VALUE });
                    return;
                }
            }
    });
}

void ParamsModel::initParamItems()
{
    auto spNode = m_wpNode.lock();
    ZASSERT_EXIT(spNode);
    std::vector<std::shared_ptr<zeno::IParam>> inputs = spNode->get_input_params();
    for (std::shared_ptr<zeno::IParam> spParam : inputs) {
        ParamItem item;
        item.bInput = true;
        item.control = spParam->control;
        if (item.control == zeno::NullControl)
            item.control = UiHelper::getDefaultControl(spParam->type);
        item.optCtrlprops = spParam->optCtrlprops;
        item.m_wpParam = spParam;
        item.name = QString::fromStdString(spParam->name);
        item.type = spParam->type;
        item.value = UiHelper::zvarToQVar(spParam->defl);
        item.connectProp = spParam->socketType;
        m_items.append(item);
    }

    std::vector<std::shared_ptr<zeno::IParam>> outputs = spNode->get_output_params();
    for (std::shared_ptr<zeno::IParam> spParam : outputs) {
        ParamItem item;
        item.bInput = false;
        item.m_wpParam = spParam;
        item.control = zeno::NullControl;
        item.name = QString::fromStdString(spParam->name);
        item.type = spParam->type;
        item.connectProp = spParam->socketType;
        m_items.append(item);
    }
}

QVariant ParamsModel::data(const QModelIndex& index, int role) const
{
    const ParamItem& param = m_items[index.row()];

    switch (role)
    {
    case ROLE_PARAM_NAME:       return param.name;
    case ROLE_PARAM_TYPE:       return param.type;
    case ROLE_PARAM_VALUE:      return param.value;
    case ROLE_PARAM_CONTROL:    return param.control;
    case ROLE_SOCKET_TYPE:return param.connectProp;
    case ROLE_ISINPUT:          return param.bInput;
    case ROLE_NODEIDX:          return m_nodeIdx;
    case ROLE_LINKS:            return QVariant::fromValue(param.links);
    case ROLE_PARAM_SOCKPROP: {
        //TODO: based on core data `ParamInfo.prop`
        break;
    }
    case ROLE_PARAM_CTRL_PROPERTIES: {
        if (param.optCtrlprops.has_value())
            return QVariant::fromValue(param.optCtrlprops.value());
        else
            return QVariant();
    }
    case ROLE_PARAM_INFO: {
        zeno::ParamInfo info;
        info.name = param.name.toStdString();
        info.type = param.type;
        info.control = param.control;
        info.ctrlProps = param.optCtrlprops;
        info.defl = UiHelper::qvarToZVar(param.value, info.type);
        info.socketType = param.connectProp;
        for (auto linkidx : param.links) {
            info.links.push_back(linkidx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>());
        }
        return QVariant::fromValue(info);
    }
    case ROLE_NODE_IDX:
    {
        return m_nodeIdx;
    }
    case ROLE_NODE_NAME:
    {
        return m_nodeIdx.data(ROLE_NODE_NAME);
    }
    }
    return QVariant();
}

int ParamsModel::indexFromName(const QString& name, bool bInput) const
{
    for (int i = 0; i < m_items.length(); i++) {
        if (m_items[i].name == name && m_items[i].bInput == bInput) {
            return i;
        }
    }
    return -1;
}

QVariant ParamsModel::getIndexList(bool bInput) const
{
    QVariantList varList;
    for (int i = 0; i < m_items.length(); i++) {
        if (m_items[i].bInput == bInput) {
            varList.append(i);
        }
    }
    return varList;
}

GraphModel* ParamsModel::getGraph() const
{
    if (NodeItem* pItem = qobject_cast<NodeItem*>(parent())) {
        if (GraphModel* pModel = qobject_cast<GraphModel*>(pItem->parent())) {
            return pModel;
        }
    }
    return nullptr;
}

PARAMS_INFO ParamsModel::getInputs()
{
    PARAMS_INFO params_inputs;
    for (ParamItem& item : m_items)
    {
        if (item.bInput)
        {
            zeno::ParamInfo info;
            info.name = item.name.toStdString();
            info.type = item.type;
            info.control = item.control;
            info.ctrlProps = item.optCtrlprops;
            info.defl = UiHelper::qvarToZVar(item.value, info.type);
            info.socketType = item.connectProp;
            for (auto linkidx : item.links) {
                info.links.push_back(linkidx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>());
            }
            //info.tooltip //std::string tooltip;
            //info.prop   //SocketProperty prop = Socket_Normal;
            params_inputs.insert(item.name, info);
        }
    }
    return params_inputs;
}

PARAMS_INFO ParamsModel::getOutputs()
{
    PARAMS_INFO params_outputs;
    for (ParamItem& item : m_items)
    {
        if (!item.bInput)
        {
            zeno::ParamInfo info;
            info.name = item.name.toStdString();
            info.type = item.type;
            info.control = item.control;
            info.ctrlProps = item.optCtrlprops;
            info.defl = UiHelper::qvarToZVar(item.value, info.type);
            info.socketType = item.connectProp;
            for (auto linkidx : item.links) {
                info.links.push_back(linkidx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>());
            }
            //info.tooltip //std::string tooltip;
            //info.prop   //SocketProperty prop = Socket_Normal;
            params_outputs.insert(item.name, info);
        }
    }
    return params_outputs;
}

bool ParamsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    ParamItem& param = m_items[index.row()];
    switch (role) {
    case ROLE_PARAM_NAME:
        //TODO: update param name to coredata
        param.name = value.toString();
        break;

    case ROLE_PARAM_TYPE:
        param.type = (zeno::ParamType)value.toInt();
        break;

    case ROLE_PARAM_VALUE:
    {
        auto spNode = m_wpNode.lock();
        if (spNode) {
            zeno::zvariant defl = UiHelper::qvarToZVar(value, param.type);
            spNode->update_param(param.name.toStdString(), defl);
            return true;        //the dataChanged signal will be emitted by registered callback function.
        }
        return false;
    }

    case ROLE_PARAM_CONTROL:
        param.control = (zeno::ParamControl)value.toInt();
        break;
    case ROLE_PARAM_CTRL_PROPERTIES:
        param.optCtrlprops = value.value<zeno::ControlProperty>();
        break;
    default:
        return false;
    }

    emit dataChanged(index, index, QVector<int>{role});
    return true;
}

QHash<int, QByteArray> ParamsModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ROLE_PARAM_NAME] = "name";
    roles[ROLE_PARAM_TYPE] = "type";
    roles[ROLE_PARAM_CONTROL] = "control";
    roles[ROLE_ISINPUT] = "input";
    return roles;
}

int ParamsModel::rowCount(const QModelIndex& parent) const
{
    return m_items.count();
}

void ParamsModel::setNodeIdx(const QModelIndex& nodeIdx)
{
    m_nodeIdx = nodeIdx;
}

QModelIndex ParamsModel::paramIdx(const QString& name, bool bInput) const
{
    for (int r = 0; r < rowCount(); r++)
    {
        QModelIndex idx = index(r, 0);
        if (name == data(idx, ROLE_PARAM_NAME).toString() && bInput == data(idx, ROLE_ISINPUT).toBool())
            return idx;
    }
    return QModelIndex();
}

void ParamsModel::addLink(const QModelIndex& paramIdx, const QPersistentModelIndex& linkIdx)
{
    m_items[paramIdx.row()].links.append(linkIdx);
}

int ParamsModel::removeLink(const QModelIndex& paramIdx)
{
    QList<QPersistentModelIndex>& links = m_items[paramIdx.row()].links;
    if (links.isEmpty())
        return -1;

    //ZASSERT_EXIT(links.size() == 1);
    int nRow = links[0].row();
    links.clear();
    return nRow;
}

bool ParamsModel::removeSpecificLink(const QModelIndex& paramIdx, const QModelIndex& linkIdx)
{
    int row = paramIdx.row();
    if (row < 0 || row >= m_items.size())
        return false;

    QList<QPersistentModelIndex>& links = m_items[row].links;
    for (auto link : links) {
        if (link == linkIdx)
            return true;
    }
    return false;
}

QModelIndex ParamsModel::removeOneLink(const QModelIndex& paramIdx, const zeno::EdgeInfo& link)
{
    QList<QPersistentModelIndex>& links = m_items[paramIdx.row()].links;
    if (links.isEmpty())
        return QModelIndex();

    for (auto it = links.begin(); it != links.end(); it++) {
        const zeno::EdgeInfo& lnk = (*it).data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        if (lnk == link) {
            QModelIndex idx = *it;
            it = links.erase(it);
            return idx;
        }
    }
    return QModelIndex();
}

void ParamsModel::addParam(const ParamItem& param)
{
    int nRows = m_items.size();
    beginInsertRows(QModelIndex(), nRows, nRows);
    m_items.append(param);
    endInsertRows();
}

GraphModel* ParamsModel::parentGraph() const
{
    if (auto pNode = qobject_cast<NodeItem*>(parent())) {
        return qobject_cast<GraphModel*>(pNode->parent());
    }
    return nullptr;
}

QStandardItemModel* ParamsModel::customParamModel()
{
    if (!m_customParamsM)
    {
        m_customParamsM = new QStandardItemModel(this);
        resetCustomParamModel();
    }
    
    return m_customParamsM;
}

void ParamsModel::batchModifyParams(const zeno::ParamsUpdateInfo& params)
{
    if (params.empty())
        return;

    auto spNode = m_wpNode.lock();
    ZASSERT_EXIT(spNode);
    zeno::params_change_info changes = spNode->update_editparams(params);

    //assuming that the param layout has changed, and we muse reconstruct all params and index.
    emit layoutAboutToBeChanged();

    //remove old links from this node.
    for (int r = 0; r < m_items.size(); r++) {
        ParamItem& item = m_items[r];
        for (QPersistentModelIndex linkIdx : item.links) {
            if (item.bInput) {
                QModelIndex outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
                //only remove link by model itself, with no action about core data.
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(outSockIdx.model());
                ParamsModel* outParams = qobject_cast<ParamsModel*>(pModel);
                ZASSERT_EXIT(outParams);
                bool ret = outParams->removeSpecificLink(outSockIdx, linkIdx);
                ZASSERT_EXIT(ret);
            }
            else {
                QModelIndex inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
                //only remove link by model itself, with no action about core data.
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(inSockIdx.model());
                ParamsModel* inParams = qobject_cast<ParamsModel*>(pModel);
                ZASSERT_EXIT(inParams);
                bool ret = inParams->removeSpecificLink(inSockIdx, linkIdx);
                ZASSERT_EXIT(ret);
            }
        }

        for (QPersistentModelIndex linkIdx : item.links) {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(linkIdx.model());
            LinkModel* linkModel = qobject_cast<LinkModel*>(pModel);
            ZASSERT_EXIT(linkModel);
            //no standard api to user, just delete from model, and sync to ui.
            linkModel->removeRows(linkIdx.row(), 1);
        }

        item.links.clear();
    }

    m_items.clear();
    //reconstruct params.
    initParamItems();

    //reconstruct links.
    for (int r = 0; r < m_items.size(); r++) {
        std::shared_ptr<zeno::IParam> spParam = m_items[r].m_wpParam.lock();
        for (auto spLink : spParam->links) {
            auto spFrom = spLink->fromparam.lock();
            auto spTo = spLink->toparam.lock();
            ZASSERT_EXIT(spFrom && spTo);
            auto spFromNode = spFrom->m_wpNode.lock();
            auto spToNode = spTo->m_wpNode.lock();
            ZASSERT_EXIT(spFromNode && spToNode);

            const QString fromNode = QString::fromStdString(spFromNode->get_name());
            const QString toNode = QString::fromStdString(spToNode->get_name());
            const QString fromSock = QString::fromStdString(spFrom->name);
            const QString toSock = QString::fromStdString(spTo->name);
            const QString outKey = QString::fromStdString(spLink->tokey);
            const QString inKey = QString::fromStdString(spLink->fromkey);
            zeno::LinkFunction lnkProp = spLink->lnkProp;

            //add the new link in current graph.
            GraphModel* pGraphM = parentGraph();
            QModelIndex fromNodeIdx = pGraphM->indexFromName(fromNode);
            QModelIndex toNodeIdx = pGraphM->indexFromName(toNode);
            ZASSERT_EXIT(fromNodeIdx.isValid() && toNodeIdx.isValid());

            ParamsModel* fromParams = QVariantPtr<ParamsModel>::asPtr(fromNodeIdx.data(ROLE_PARAMS));
            ParamsModel* toParams = QVariantPtr<ParamsModel>::asPtr(toNodeIdx.data(ROLE_PARAMS));
            ZASSERT_EXIT(fromParams && toParams);
            QModelIndex fromParam = fromParams->paramIdx(fromSock, false);
            QModelIndex toParam = toParams->paramIdx(toSock, true);
            ZASSERT_EXIT(fromParam.isValid() && toParam.isValid());

            LinkModel* lnkModel = pGraphM->getLinkModel();
            ZASSERT_EXIT(lnkModel);
            //only add in model layer, not core layer.
            QModelIndex newLink = lnkModel->addLink(fromParam, outKey, toParam, inKey, lnkProp);

            fromParams->m_items[fromParam.row()].links.append(newLink);
            toParams->m_items[toParam.row()].links.append(newLink);
        }
    }
    resetCustomParamModel();
    emit layoutChanged();
}

void ParamsModel::resetCustomParamModel()
{
    if (!m_customParamsM)
        return;
    m_customParamsM->clear();
    QStandardItem* pRoot = new QStandardItem("root");
    QStandardItem* pInputs = new QStandardItem("input");
    QStandardItem* pOutputs = new QStandardItem("output");
    for (int r = 0; r < rowCount(); r++) {
        QModelIndex paramIdx = index(r);
        bool bInput = paramIdx.data(ROLE_ISINPUT).toBool();
        zeno::ParamInfo info = paramIdx.data(ROLE_PARAM_INFO).value<zeno::ParamInfo>();
        QStandardItem* paramItem = new QStandardItem(QString::fromStdString(info.name));
        const QString& paramName = QString::fromStdString(info.name);
        paramItem->setData(paramName, Qt::DisplayRole);
        paramItem->setData(paramName, ROLE_PARAM_NAME);
        paramItem->setData(paramName, ROLE_MAP_TO_PARAMNAME);
        paramItem->setData(UiHelper::zvarToQVar(info.defl), ROLE_PARAM_VALUE);
        paramItem->setData(info.control, ROLE_PARAM_CONTROL);
        paramItem->setData(info.type, ROLE_PARAM_TYPE);
        paramItem->setData(bInput, ROLE_ISINPUT);
        paramItem->setData(info.socketType, ROLE_SOCKET_TYPE);
        if (info.ctrlProps.has_value())
            paramItem->setData(QVariant::fromValue(info.ctrlProps.value()), ROLE_PARAM_CTRL_PROPERTIES);
        if (bInput)
            pInputs->appendRow(paramItem);
        else
            pOutputs->appendRow(paramItem);
        paramItem->setData(VPARAM_PARAM, ROLE_ELEMENT_TYPE);
        paramItem->setEditable(true);
    }
    pRoot->setEditable(false);
    pRoot->setData(VPARAM_TAB, ROLE_ELEMENT_TYPE);
    pInputs->setEditable(false);
    pInputs->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);
    pOutputs->setEditable(false);
    pOutputs->setData(VPARAM_GROUP, ROLE_ELEMENT_TYPE);

    pRoot->appendRow(pInputs);
    pRoot->appendRow(pOutputs);

    m_customParamsM->appendRow(pRoot);
}

bool ParamsModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    m_items.removeAt(row);
    endRemoveRows();
    return true;
}

int ParamsModel::getParamlinkCount(const QModelIndex& paramIdx)
{
    return m_items[paramIdx.row()].links.size();
}

int ParamsModel::numOfInputParams() const
{
    int n = 0;
    for (auto item : m_items) {
        if (item.bInput)
            n++;
    }
    return n;
}
