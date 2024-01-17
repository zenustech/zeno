#include "parammodel.h"
#include "zassert.h"
#include "util/uihelper.h"
#include <zeno/core/data.h>
#include <zeno/core/IParam.h>


ParamsModel::ParamsModel(std::shared_ptr<zeno::INode> spNode, QObject* parent)
    : QAbstractListModel(parent)
    , m_wpNode(spNode)
{
    std::vector<std::shared_ptr<zeno::IParam>> inputs = spNode->get_input_params();
    for (std::shared_ptr<zeno::IParam> spParam : inputs) {
        ParamItem item;
        item.bInput = true;
        item.control = UiHelper::getDefaultControl(spParam->type);
        item.m_wpParam = spParam;
        item.name = QString::fromStdString(spParam->name);
        item.type = spParam->type;
        item.value = UiHelper::zvarToQVar(spParam->defl);
        m_items.append(item);
    }

    std::vector<std::shared_ptr<zeno::IParam>> outputs = spNode->get_output_params();
    for (std::shared_ptr<zeno::IParam> spParam : outputs) {
        ParamItem item;
        item.bInput = false;
        item.m_wpParam = spParam;
        item.name = QString::fromStdString(spParam->name);
        item.type = spParam->type;
        m_items.append(item);
    }

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

QVariant ParamsModel::data(const QModelIndex& index, int role) const
{
    const ParamItem& param = m_items[index.row()];

    switch (role)
    {
    case ROLE_PARAM_NAME:       return param.name;
    case ROLE_PARAM_TYPE:       return param.type;
    case ROLE_PARAM_VALUE:      return param.value;
    case ROLE_PARAM_CONTROL:    return param.control;
    case ROLE_ISINPUT:          return param.bInput;
    case ROLE_NODEIDX:          return m_nodeIdx;
    case ROLE_LINKS:            return QVariant::fromValue(param.links);
    case ROLE_PARAM_SOCKPROP: {
        //TODO: based on core data `ParamInfo.prop`
        break;
    }
    case ROLE_PARAM_CTRL_PROPERTIES: {
        //TODO: control property
        break;
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

void ParamsModel::addParam(const ParamItem& param)
{
    int nRows = m_items.size();
    beginInsertRows(QModelIndex(), nRows, nRows);
    m_items.append(param);
    endInsertRows();
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
