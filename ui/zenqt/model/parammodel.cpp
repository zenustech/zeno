#include "parammodel.h"
#include "zassert.h"
#include "util/uihelper.h"
#include <zeno/core/data.h>
#include <zeno/core/CoreParam.h>
#include "model/LinkModel.h"
#include "model/GraphModel.h"
#include "variantptr.h"
#include "model/graphsmanager.h"
#include "model/graphstreemodel.h"


ParamsModel::ParamsModel(std::shared_ptr<zeno::INode> spNode, QObject* parent)
    : QAbstractListModel(parent)
    , m_wpNode(spNode)
    , m_customParamsM(nullptr)
{
    initParamItems();

    //TODO: register callback for core param adding/removing, for the functionally of custom param panel.
    cbUpdateParam = spNode->register_update_param(
        [this](const std::string& name, zeno::reflect::Any old_value, zeno::reflect::Any new_value) {
            QVariant newValue = QVariant::fromValue(new_value);
            for (int i = 0; i < m_items.size(); i++) {
                if (m_items[i].name.toStdString() == name) {
                    m_items[i].value = newValue; //update cache
                    QModelIndex idx = createIndex(i, 0);
                    emit dataChanged(idx, idx, { ROLE_PARAM_VALUE });
                    break;
                }
            }
            Qt::MatchFlags flags = Qt::MatchRecursive | Qt::MatchCaseSensitive;
            auto pItems = m_customParamsM->findItems(QString::fromStdString(name), flags);
            for (auto pItem : pItems)
            {
                auto oldVal = pItem->data(ROLE_PARAM_VALUE);
                if (pItem->data(ROLE_ISINPUT).toBool())
                {
                    if (newValue.type() == QVariant::UserType)
                    {
                        QString newStr = UiHelper::variantToString(newValue);
                        QString oldStr = UiHelper::variantToString(oldVal);
                        if (oldStr == newStr)
                            continue;
                    }
                    pItem->setData(newValue, ROLE_PARAM_VALUE);
                }
            }
    });

    spNode->register_update_param_socket_type(
        [this](const std::string& name, zeno::SocketType type) {
        updateParamData(QString::fromStdString(name), type, ROLE_SOCKET_TYPE);
    });

    spNode->register_update_param_type(
        [this](const std::string& name, zeno::ParamType type) {
        updateParamData(QString::fromStdString(name), type, ROLE_PARAM_TYPE);
    });

    spNode->register_update_param_control(
        [this](const std::string& name, zeno::ParamControl control) {
        updateParamData(QString::fromStdString(name), control, ROLE_PARAM_CONTROL);
    });

    spNode->register_update_param_control_prop(
        [this](const std::string& name, zeno::ControlProperty controlProps) {
        updateParamData(QString::fromStdString(name), QVariant::fromValue(controlProps), ROLE_PARAM_CTRL_PROPERTIES);
    });

    spNode->register_update_param_visible(
        [this](const std::string& name, bool bVisible) {
        updateParamData(QString::fromStdString(name), bVisible, ROLE_PARAM_VISIBLE);
    });
}

void ParamsModel::initParamItems()
{
    auto spNode = m_wpNode.lock();
    ZASSERT_EXIT(spNode);
    //primitive inputs
    auto inputs = spNode->get_input_primitive_params();
    for (const auto& spParam : inputs) {
        ParamItem item;
        item.bInput = true;
        item.control = spParam.control;
        if (item.control == zeno::NullControl)
            item.control = UiHelper::getDefaultControl(spParam.type);
        item.optCtrlprops = spParam.ctrlProps;
        item.name = QString::fromStdString(spParam.name);
        item.type = spParam.type;
        item.value = QVariant::fromValue(spParam.defl);
        item.connectProp = spParam.socketType;
        item.bVisible = spParam.bVisible;
        item.group = zeno::Group_InputPrimitive;
        m_items.append(item);
    }
    //object inputs
    auto objInputs = spNode->get_input_object_params();
    for (const auto& spParam : objInputs) {
        ParamItem item;
        item.bInput = true;
        item.name = QString::fromStdString(spParam.name);
        item.type = spParam.type;
        item.connectProp = spParam.socketType;
        item.group = zeno::Group_InputObject;
        m_items.append(item);
    }

    //primitive outputs
    auto outputs = spNode->get_output_primitive_params();
    for (const auto& param : outputs) {
        ParamItem item;
        item.bInput = false;
        item.control = zeno::NullControl;
        item.name = QString::fromStdString(param.name);
        item.type = param.type;
        item.connectProp = param.socketType;
        item.group = zeno::Group_OutputPrimitive;
        m_items.append(item);
    }

    //object outputs
    auto objOutputs = spNode->get_output_object_params();
    for (const auto& param : objOutputs) {
        ParamItem item;
        item.bInput = false;
        item.name = QString::fromStdString(param.name);
        item.type = param.type;
        item.connectProp = param.socketType;
        item.group = zeno::Group_OutputObject;
        m_items.append(item);
    }

    //init custom param model.
    initCustomUI(spNode->get_customui());
}

void ParamsModel::initCustomUI(const zeno::CustomUI& customui)
{
    if (m_customParamsM) {
        m_customParamsM->clear();
    }
    else {
        m_customParamsM = new QStandardItemModel(this);
        connect(m_customParamsM, &QStandardItemModel::dataChanged, this, &ParamsModel::onCustomModelDataChanged);
    }
    UiHelper::newCustomModel(m_customParamsM, customui);

    //m_customParamsM创建后需更新初始值
    QStandardItem* pInputsRoot = m_customParamsM->item(0);
    for (int i = 0; i < pInputsRoot->rowCount(); i++)
    {
        auto tabItem = pInputsRoot->child(i);
        for (int j = 0; j < tabItem->rowCount(); j++)
        {
            auto groupItem = tabItem->child(j);
            for (int k = 0; k < groupItem->rowCount(); k++)
            {
                auto paramItem = groupItem->child(k);
                int row = indexFromName(paramItem->data(ROLE_PARAM_NAME).toString(), true);
                if (row != -1)
                {
                    paramItem->setData(m_items[row].value, ROLE_PARAM_VALUE);
                    paramItem->setData(m_items[row].bVisible, ROLE_PARAM_VISIBLE);
                }
            }
        }
    }
}

void ParamsModel::onCustomModelDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    for (int role : roles)
    {
        if (role != ROLE_PARAM_VALUE)
            continue;

        QVariant newValue = topLeft.data(ROLE_PARAM_VALUE);
        const zeno::ParamType type = (zeno::ParamType)topLeft.data(ROLE_PARAM_TYPE).toInt();
        QString name = topLeft.data(ROLE_PARAM_NAME).toString();
        
        auto spNode = m_wpNode.lock();
        if (spNode) {
            const auto& defl = newValue.value<zeno::reflect::Any>();
            bool bOldEntry = m_bReentry;
            zeno::scope_exit scope([&]() { m_bReentry = bOldEntry; });
            m_bReentry = true;
            spNode->update_param(name.toStdString(), defl);
        }
    }
}

bool ParamsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    ParamItem& param = m_items[index.row()];
    switch (role) {
    case ROLE_PARAM_NAME:
        param.name = value.toString();
        break;

    case ROLE_PARAM_TYPE:
        param.type = (zeno::ParamType)value.toInt();
        break;

    case ROLE_PARAM_VALUE:
    {
        auto spNode = m_wpNode.lock();
        if (spNode) {
            const auto& defl = UiHelper::qvarToAny(value);
            assert(defl.has_value());
            spNode->update_param(param.name.toStdString(), defl);
            GraphsManager::instance().currentModel()->markDirty(true);
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
    case ROLE_SOCKET_TYPE:
    {
        auto spNode = m_wpNode.lock();
        if (spNode) {
            param.connectProp = (zeno::SocketType)value.toInt();
            spNode->update_param_socket_type(param.name.toStdString(), param.connectProp);
            return true;
        }
        return false;
    }
    case ROLE_PARAM_VISIBLE:
    {
        auto spNode = m_wpNode.lock();
        if (spNode) {
            spNode->update_param_visible(param.name.toStdString(), value.toBool());
            return true;
        }
        return false;
    }
    default:
        return false;
    }

    emit dataChanged(index, index, QVector<int>{role});
    GraphsManager::instance().currentModel()->markDirty(true);
    return true;
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
    case ROLE_SOCKET_TYPE:      return param.connectProp;
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
        zeno::ParamPrimitive info;
        info.name = param.name.toStdString();
        info.type = param.type;
        info.control = param.control;
        info.ctrlProps = param.optCtrlprops;
        info.defl = param.value.value<zeno::reflect::Any>();
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
    case ROLE_PARAM_VISIBLE:
        return param.bVisible;
    case ROLE_PARAM_GROUP:
        return param.group;
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
            zeno::ParamPrimitive info;
            info.name = item.name.toStdString();
            info.type = item.type;
            info.control = item.control;
            info.ctrlProps = item.optCtrlprops;
            info.defl = item.value.value<zeno::reflect::Any>();
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
            zeno::ParamPrimitive info;
            info.name = item.name.toStdString();
            info.type = item.type;
            info.control = item.control;
            info.ctrlProps = item.optCtrlprops;
            info.defl = item.value.value<zeno::reflect::Any>();
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
    return m_customParamsM;
}

void ParamsModel::batchModifyParams(const zeno::ParamsUpdateInfo& params)
{
    //if (params.empty())   //可能是删除到空的情况，无需return
    //    return;

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
        auto group = m_items[r].group;
        std::vector<zeno::EdgeInfo> links;
        if (group == zeno::Group_InputPrimitive)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_input_prim_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;
            
        }
        else if (group == zeno::Group_InputObject)
        {
            bool bExist = false;
            auto paramObj = spNode->get_input_obj_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramObj.links;
        }
        for (const auto& linkInfo : links) {

            const QString fromNode = QString::fromStdString(linkInfo.outNode);
            const QString toNode = QString::fromStdString(linkInfo.inNode);
            const QString fromSock = QString::fromStdString(linkInfo.outParam);
            const QString toSock = QString::fromStdString(linkInfo.inParam);
            const QString outKey = QString::fromStdString(linkInfo.outKey);
            const QString inKey = QString::fromStdString(linkInfo.inKey);

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
            QModelIndex newLink = lnkModel->addLink(fromParam, outKey, toParam, inKey, linkInfo.bObjLink);

            fromParams->m_items[fromParam.row()].links.append(newLink);
            toParams->m_items[toParam.row()].links.append(newLink);
        }
    }
    //resetCustomParamModel();
    emit layoutChanged();
}

void ParamsModel::test_customparamsmodel() const
{
    QStandardItem* pRoot = m_customParamsM->invisibleRootItem();
    for (int i = 0; i < pRoot->rowCount(); i++)
    {
        QStandardItem* pItem = pRoot->child(i);
        QString wtf = pItem->text();
        for (int j = 0; j < pItem->rowCount(); j++)
        {
            QStandardItem* ppItem = pItem->child(j);
            wtf = ppItem->text();
            for (int k = 0; k < ppItem->rowCount(); k++)
            {
                QStandardItem* pppItem = ppItem->child(k);
                wtf = pppItem->text();
            }
        }
    }
}

void ParamsModel::updateParamData(const QString& name, const QVariant& val, int role)
{
    for (int i = 0; i < m_items.size(); i++) {
        if (m_items[i].name == name) {
            if (role == ROLE_PARAM_CONTROL)
                m_items[i].control = (zeno::ParamControl)val.toInt();
            else if (role == ROLE_PARAM_TYPE)
                m_items[i].type = (zeno::ParamType)val.toInt();
            else if (role == ROLE_SOCKET_TYPE)
                m_items[i].connectProp = (zeno::SocketType)val.toInt();
            else if (role == ROLE_PARAM_CTRL_PROPERTIES)
                m_items[i].optCtrlprops = val.value<zeno::ControlProperty>();
            else if (role == ROLE_PARAM_VISIBLE)
                m_items[i].bVisible = val.toBool();
            else if (role == ROLE_PARAM_GROUP)
                m_items[i].group = (zeno::NodeDataGroup)val.toInt();
            QModelIndex idx = createIndex(i, 0);
            emit dataChanged(idx, idx, { role });
            break;
        }
    }
    //object inputs do not need to update custom model
    if (role == ROLE_SOCKET_TYPE || role == ROLE_PARAM_GROUP)
        return;
    Qt::MatchFlags flags = Qt::MatchRecursive | Qt::MatchCaseSensitive;
    auto pItems = m_customParamsM->findItems(name, flags);
    for (auto pItem : pItems)
    {
        if (pItem->data(ROLE_ISINPUT).toBool())
        {
            pItem->setData(val, role);
        }
    }
}

void ParamsModel::resetCustomUi(const zeno::CustomUI& customui)
{
    auto spNode = m_wpNode.lock();
    if (std::shared_ptr<zeno::SubnetNode> sbn = std::dynamic_pointer_cast<zeno::SubnetNode>(spNode))
        sbn->setCustomUi(customui);
}

bool ParamsModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    m_items.removeAt(row);
    endRemoveRows();
    return true;
}

void ParamsModel::getDegrees(int& inDegrees, int& outDegrees) {
    inDegrees = outDegrees = 0;
    for (auto item : m_items) {
        if (item.bInput) {
            inDegrees += item.links.size();
        }
        else {
            outDegrees += item.links.size();
        }
    }
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
