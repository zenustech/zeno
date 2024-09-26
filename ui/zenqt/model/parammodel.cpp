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
#include <zeno/utils/helper.h>


class CustomUIProxyModel : public QStandardItemModel
{
public:
    CustomUIProxyModel(ParamsModel* parent = nullptr) : QStandardItemModel(parent), m_baseM(parent) {}
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override {
        if (role == ROLE_PARAM_VALUE) {
            QString paramName = index.data(ROLE_PARAM_NAME).toString();
            QModelIndex idxparam = m_baseM->paramIdx(paramName, true);
            QVariant wtf = idxparam.data(ROLE_PARAM_VALUE);
            return wtf;
        }
        else {
            return QStandardItemModel::data(index, role);
        }
    }

private:
    ParamsModel* m_baseM;
};


ParamsModel::ParamsModel(std::shared_ptr<zeno::INode> spNode, QObject* parent)
    : QAbstractListModel(parent)
    , m_wpNode(spNode)
    , m_customParamsM(nullptr)
{
    initParamItems();
    initCustomUI(spNode->export_customui());

    cbUpdateParam = spNode->register_update_param(
        [this](const std::string& name, zeno::reflect::Any old_value, zeno::reflect::Any new_value) {
            for (int i = 0; i < m_items.size(); i++) {
                if (m_items[i].name.toStdString() == name) {
                    if (m_items[i].value != new_value)
                    {
                        QModelIndex idx = createIndex(i, 0);
                        setData(idx, QVariant::fromValue(new_value), ROLE_PARAM_VALUE);
                    }
                    break;
                }
            }
            //根据需要更新节点布局
            auto spNode = m_wpNode.lock();
            ZASSERT_EXIT(spNode);
            spNode->trigger_update_params(name, false, zeno::params_change_info());
        });

    spNode->register_update_param_socket_type(
        [this](const std::string& name, zeno::SocketType type) {
            updateParamData(QString::fromStdString(name), type, ROLE_SOCKET_TYPE);
        });

    spNode->register_update_param_type(
        [this](const std::string& name, zeno::ParamType type, bool bInput) {
        updateParamData(QString::fromStdString(name), type, ROLE_PARAM_TYPE, bInput);
        });

    spNode->register_update_param_control(
        [this](const std::string& name, zeno::ParamControl control) {
            updateParamData(QString::fromStdString(name), control, ROLE_PARAM_CONTROL);
        });

    spNode->register_update_param_control_prop(
        [this](const std::string& name, zeno::reflect::Any controlProps) {
            updateParamData(QString::fromStdString(name), QVariant::fromValue(controlProps), ROLE_PARAM_CTRL_PROPERTIES);
        });

    spNode->register_update_param_socket_visible(
        [this](const std::string& name, bool bSocketVisible, bool bInput) {
            updateParamData(QString::fromStdString(name), bSocketVisible, ROLE_PARAM_SOCKET_VISIBLE, bInput);
        });

    spNode->register_update_visable_enable([this](zeno::INode* pNode, std::set<std::string> adjInputs, std::set<std::string> adjOutputs) {
        //扫一遍，更新一下缓存值
        for (ParamItem& item : m_items) {
            std::string name = item.name.toStdString();
            if (adjInputs.find(name) != adjInputs.end() && item.bInput) {
                bool bExist = false;
                zeno::CommonParam param = pNode->get_input_param(name, &bExist);
                ZASSERT_EXIT(bExist);
                if (param.bEnable != item.bEnable) {
                    updateParamData(item.name, param.bEnable, ROLE_PARAM_ENABLE, true);
                }
                if (param.bVisible != item.bVisible) {
                    updateParamData(item.name, param.bVisible, ROLE_PARAM_VISIBLE, true);
                }
            }
            if (adjOutputs.find(name) != adjOutputs.end() && !item.bInput) {
                bool bExist = false;
                zeno::CommonParam param = pNode->get_output_param(name, &bExist);
                ZASSERT_EXIT(bExist);
                if (param.bEnable != item.bEnable) {
                    updateParamData(item.name, param.bEnable, ROLE_PARAM_ENABLE, false);
                }
                if (param.bVisible != item.bVisible) {
                    updateParamData(item.name, param.bVisible, ROLE_PARAM_VISIBLE, false);
                }
            }
        }
        emit enabledVisibleChanged();
    });

    spNode->register_update_param_color(
        [this](const std::string& name, std::string& clr) {
            updateParamData(QString::fromStdString(name), QString::fromStdString(clr), ROLE_PARAM_SOCKET_CLR);
        });

    spNode->register_update_layout(
        [this](zeno::params_change_info& changes) {
            updateUiLinksSockets(changes);
        });
}

void ParamsModel::initParamItems()
{
    auto spNode = m_wpNode.lock();
    ZASSERT_EXIT(spNode);
    //primitive inputs
    const zeno::CustomUI& customui = spNode->export_customui();
    if (!customui.inputPrims.empty() && !customui.inputPrims[0].groups.empty()) {
        auto inputs = customui.inputPrims[0].groups[0].params;
        for (const zeno::ParamPrimitive& spParam : inputs) {
            ParamItem item;
            item.bInput = true;
            item.control = spParam.control;
            if (item.control == zeno::NullControl)
                item.control = zeno::getDefaultControl(spParam.type);
            item.optCtrlprops = spParam.ctrlProps;
            item.name = QString::fromStdString(spParam.name);
            item.type = spParam.type;
            item.value = spParam.defl;
            item.connectProp = spParam.socketType;
            item.bSocketVisible = spParam.bSocketVisible;
            item.bVisible = spParam.bVisible;
            item.bEnable = spParam.bEnable;
            item.group = zeno::Role_InputPrimitive;
            item.sockProp = spParam.sockProp;
            m_items.append(item);
        }
    }
    //object inputs
    for (const auto& spParam : customui.inputObjs) {
        ParamItem item;
        item.bInput = true;
        item.name = QString::fromStdString(spParam.name);
        item.type = spParam.type;
        item.connectProp = spParam.socketType;
        item.group = zeno::Role_InputObject;
        item.bVisible = spParam.bVisible;
        item.bEnable = spParam.bEnable;
        m_items.append(item);
    }

    //primitive outputs
    for (const auto& param : customui.outputPrims) {
        ParamItem item;
        item.bInput = false;
        item.control = zeno::NullControl;
        item.name = QString::fromStdString(param.name);
        item.type = param.type;
        item.connectProp = param.socketType;
        item.group = zeno::Role_OutputPrimitive;
        item.bSocketVisible = param.bSocketVisible;
        item.bVisible = param.bVisible;
        item.bEnable = param.bEnable;
        m_items.append(item);
    }

    //object outputs
    for (const auto& param : customui.outputObjs) {
        ParamItem item;
        item.bInput = false;
        item.name = QString::fromStdString(param.name);
        item.type = param.type;
        item.connectProp = param.socketType;
        item.group = zeno::Role_OutputObject;
        item.bVisible = param.bVisible;
        item.bEnable = param.bEnable;
        m_items.append(item);
    }

    //init custom param model.
    //initCustomUI(spNode->get_customui());
}

void ParamsModel::initCustomUI(const zeno::CustomUI& customui)
{
    if (m_customParamsM) {
        m_customParamsM->clear();
    }
    else {
        m_customParamsM = constructProxyModel();
    }
    UiHelper::newCustomModel(m_customParamsM, customui);

    //m_customParamsM创建后需更新初始值
    m_customParamsM->blockSignals(true);
    zeno::scope_exit sp([=] {m_customParamsM->blockSignals(false); });

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
                auto& paramName = paramItem->data(ROLE_PARAM_NAME).toString();
                int row = indexFromName(paramName, true);
                if (row != -1)
                {
                    paramItem->setData(QVariant::fromValue(m_items[row].value), ROLE_PARAM_VALUE);
                    paramItem->setData(m_items[row].bSocketVisible, ROLE_PARAM_SOCKET_VISIBLE);
                    paramItem->setData(m_items[row].bVisible, ROLE_PARAM_VISIBLE);
                    paramItem->setData(m_items[row].bEnable, ROLE_PARAM_ENABLE);
                }
            }
        }
    }
    QStandardItem* pOutputsRoot = m_customParamsM->item(1);
    for (int i = 0; i < pOutputsRoot->rowCount(); i++)
    {
        auto paramItem = pOutputsRoot->child(i);
        auto& paramName = paramItem->data(ROLE_PARAM_NAME).toString();
        int row = indexFromName(paramName, false);
        paramItem->setData(m_items[row].bSocketVisible, ROLE_PARAM_SOCKET_VISIBLE);
        paramItem->setData(m_items[row].bVisible, ROLE_PARAM_VISIBLE);
        paramItem->setData(m_items[row].bEnable, ROLE_PARAM_ENABLE);
    }
}

QStandardItemModel* ParamsModel::constructProxyModel()
{
    QStandardItemModel* pModel = new QStandardItemModel(this);
    connect(pModel, &QStandardItemModel::dataChanged, [=](const QModelIndex& topLeft, const QModelIndex&, const QVector<int>& roles) {
        bool bInput = topLeft.data(ROLE_ISINPUT).toBool();
        if (!bInput)
            return;

        for (int role : roles)
        {
            //if (role != ROLE_PARAM_VALUE)
            //    continue;

            QVariant newValue = topLeft.data(role);
            QString name = topLeft.data(ROLE_PARAM_NAME).toString();
            bool input = topLeft.data(ROLE_ISINPUT).toBool();
            const QModelIndex& paramIdx = this->paramIdx(name, input);

            //zeno::scope_exit sp([=] {this->blockSignals(false); });
            //this->blockSignals(true);
            setData(paramIdx, newValue, role);
        }
        });

    connect(this, &ParamsModel::dataChanged, [=](const QModelIndex& topLeft, const QModelIndex&, const QVector<int>& roles) {
        bool bInput = topLeft.data(ROLE_ISINPUT).toBool();
        if (!bInput)
            return;

        for (int role : roles)
        {
            //if (role != ROLE_PARAM_VALUE)
            //    continue;

            const QString& name = topLeft.data(ROLE_PARAM_NAME).toString();
            Qt::MatchFlags flags = Qt::MatchRecursive | Qt::MatchCaseSensitive;
            auto pItems = pModel->findItems(name, flags);
            for (auto pItem : pItems)
            {
                const QVariant& modelVal = topLeft.data(role);
                zeno::scope_exit sp([=] {pModel->blockSignals(false); });
                pModel->blockSignals(true);
                pItem->setData(modelVal, role);
            }
        }
        });
    return pModel;
}

void ParamsModel::updateCustomUiModelIncremental(const zeno::params_change_info& params, const zeno::CustomUI& customui)
{
    if (m_customParamsM) {
        UiHelper::udpateCustomModelIncremental(m_customParamsM, params, customui);
    }
    else {
        m_customParamsM = constructProxyModel();
        UiHelper::newCustomModel(m_customParamsM, customui);
    }
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
                    paramItem->setData(QVariant::fromValue(m_items[row].value), ROLE_PARAM_VALUE);
                    paramItem->setData(m_items[row].bSocketVisible, ROLE_PARAM_SOCKET_VISIBLE);
                }
            }
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
        param.type = (zeno::ParamType)value.toUInt();
        break;

    case ROLE_PARAM_VALUE:
    {
        const zeno::reflect::Any& anyVal = value.value<zeno::reflect::Any>();
        if (anyVal == param.value) {
            return false;
        }
        param.value = anyVal;
        auto spNode = m_wpNode.lock();
        if (spNode) {
            spNode->update_param(param.name.toStdString(), anyVal);
            break;
            //GraphsManager::instance().currentModel()->markDirty(true);
            //return true;        //the dataChanged signal will be emitted by registered callback function.
        }
        return false;
    }

    case ROLE_PARAM_CONTROL:
        param.control = (zeno::ParamControl)value.toInt();
        break;
    case ROLE_PARAM_CTRL_PROPERTIES:
        param.optCtrlprops = value.value<zeno::reflect::Any>();
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
    case ROLE_PARAM_SOCKET_VISIBLE:
    {
        if (param.sockProp == zeno::Socket_Disable)
            return false;
        auto spNode = m_wpNode.lock();
        if (spNode) {
            spNode->update_param_socket_visible(param.name.toStdString(), value.toBool(), param.bInput);
            return true;
        }
        return false;
    }
    case ROLE_NODE_DIRTY:
    {
        if (auto spNode = m_wpNode.lock()) {
            spNode->mark_dirty(value.toBool());
            return true;
        }
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
    case ROLE_PARAM_VALUE:      return QVariant::fromValue(param.value);
    case ROLE_PARAM_CONTROL:    return param.control;
    case ROLE_SOCKET_TYPE:      return param.connectProp;
    case ROLE_ISINPUT:          return param.bInput;
    case ROLE_NODEIDX:          return m_nodeIdx;
    case ROLE_LINKS:            return QVariant::fromValue(param.links);
    case ROLE_PARAM_SOCKPROP:   return param.sockProp;
    case ROLE_PARAM_CTRL_PROPERTIES: {
        if (param.optCtrlprops.has_value())
            return QVariant::fromValue(param.optCtrlprops);
        else
            return QVariant();
    }
    case ROLE_PARAM_INFO: {
        zeno::ParamPrimitive info;
        info.name = param.name.toStdString();
        info.type = param.type;
        info.control = param.control;
        info.ctrlProps = param.optCtrlprops;
        info.defl = param.value;
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
    {
        return param.bVisible;
    }
    case ROLE_PARAM_ENABLE:
    {
        return param.bEnable;
    }
    case ROLE_PARAM_SOCKET_VISIBLE:
    {
        if (param.sockProp == zeno::Socket_Disable)
            return false;
        return param.bSocketVisible;
    }
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
            info.defl = item.value;
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
            info.defl = item.value;
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
    updateUiLinksSockets(changes);
}

void ParamsModel::updateUiLinksSockets(zeno::params_change_info& changes)
{
    auto spNode = m_wpNode.lock();
    ZASSERT_EXIT(spNode);

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
    if (std::shared_ptr<zeno::SubnetNode> sbn = std::dynamic_pointer_cast<zeno::SubnetNode>(spNode)) {
        updateCustomUiModelIncremental(changes, sbn->get_customui());
    }
    else {
        updateCustomUiModelIncremental(changes, spNode->export_customui());
    }

    //reconstruct links.
    for (int r = 0; r < m_items.size(); r++) {
        auto group = m_items[r].group;
        std::vector<zeno::EdgeInfo> links;
        if (group == zeno::Role_InputPrimitive)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_input_prim_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;

        }
        else if (group == zeno::Role_InputObject)
        {
            bool bExist = false;
            auto paramObj = spNode->get_input_obj_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramObj.links;
        }
        else if (group == zeno::Role_OutputPrimitive)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_output_prim_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;
        }
        else if (group == zeno::Role_OutputObject)
        {
            bool bExist = false;
            auto paramPrim = spNode->get_output_obj_param(m_items[r].name.toStdString(), &bExist);
            if (!bExist)
                continue;
            links = paramPrim.links;
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

void ParamsModel::updateParamData(const QString& name, const QVariant& val, int role, bool bInput)
{
    for (int i = 0; i < m_items.size(); i++) {
        if (m_items[i].name == name && m_items[i].bInput == bInput) {
            if (role == ROLE_PARAM_CONTROL)
                m_items[i].control = (zeno::ParamControl)val.toInt();
            else if (role == ROLE_PARAM_TYPE)
                m_items[i].type = val.value<zeno::ParamType>();
            else if (role == ROLE_SOCKET_TYPE)
                m_items[i].connectProp = (zeno::SocketType)val.toInt();
            else if (role == ROLE_PARAM_CTRL_PROPERTIES)
                m_items[i].optCtrlprops = val.value<zeno::reflect::Any>();
            else if (role == ROLE_PARAM_SOCKET_VISIBLE) {
                if (m_items[i].bInput == bInput)
                    m_items[i].bSocketVisible = val.toBool();
                else
                    continue;
            }
            else if (role == ROLE_PARAM_ENABLE) {
                m_items[i].bEnable = val.toBool();
            }
            else if (role == ROLE_PARAM_VISIBLE) {
                m_items[i].bVisible = val.toBool();
            }
            else if (role == ROLE_PARAM_GROUP)
                m_items[i].group = (zeno::NodeDataGroup)val.toInt();
            else if (role == ROLE_PARAM_SOCKET_CLR) {
            }
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
        if (pItem->data(ROLE_ISINPUT).toBool() == bInput) //更新输入，或更新输入/输出的visible时,更新customUiModel
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

bool ParamsModel::hasVisiblePrimParam() const {
    for (auto item : m_items) {
        if (item.bSocketVisible &&
            (item.group == zeno::Role_InputPrimitive || item.group == zeno::Role_OutputPrimitive))
        {
            return true;
        }
    }
    return false;
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
