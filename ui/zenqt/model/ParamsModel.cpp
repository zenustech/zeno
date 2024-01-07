#include "ParamsModel.h"


ParamsModel::ParamsModel(NODE_DESCRIPTOR desc, QObject* parent)
    : QAbstractListModel(parent)
{

    for (SOCKET_DESCRIPTOR socket_desc : desc.inputs)
    {
        m_items.append({ true, socket_desc.name, socket_desc.type, socket_desc.control });
    }

    for (SOCKET_DESCRIPTOR socket_desc : desc.outputs)
    {
        m_items.append({ false, socket_desc.name, socket_desc.type });
    }
}

QVariant ParamsModel::data(const QModelIndex& index, int role) const
{
    const ParamItem& param = m_items[index.row()];

    switch (role)
    {
    case ROLE_OBJNAME:          return param.name;
    case ROLE_PARAM_TYPE:       return param.type;
    case ROLE_PARAM_CONTROL:    return param.control;
    case ROLE_ISINPUT:          return param.bInput;
    case ROLE_NODEIDX:          return m_nodeIdx;
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
    case ROLE_OBJNAME:
        param.name = value.toString();
        break;

    case ROLE_PARAM_TYPE:
        param.type = value.toString();
        break;

    case ROLE_PARAM_CONTROL:
        param.control = (ParamControl::Value)value.toInt();
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
    roles[ROLE_OBJNAME] = "name";
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
        if (name == data(idx, ROLE_OBJNAME).toString() && bInput == data(idx, ROLE_ISINPUT).toBool())
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
