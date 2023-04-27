#include "linkmodel.h"
#include "modelrole.h"
#include <zenomodel/include/uihelper.h>


LinkModel::LinkModel(QObject* parent)
    : QAbstractItemModel(parent)
{

}

LinkModel::~LinkModel()
{
}

QModelIndex LinkModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= m_items.size())
        return QModelIndex();

    return createIndex(row, 0, nullptr);
}

QModelIndex LinkModel::parent(const QModelIndex& child) const
{
    return QModelIndex();
}

int LinkModel::rowCount(const QModelIndex& parent) const
{
    return m_items.size();
}

int LinkModel::columnCount(const QModelIndex& parent) const
{
    return 1;
}

bool LinkModel::hasChildren(const QModelIndex& parent) const
{
    return false;
}

QVariant LinkModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    int r = index.row();
    if (r < 0 || r >= m_items.size())
        return QVariant();

    const QModelIndex& fromSock = m_items[r].fromSock;
    const QModelIndex& toSock = m_items[r].toSock;

    switch (role)
    {
    case ROLE_OBJID:   return m_items[r].ident;
    case ROLE_OUTNODE:
    {
        if (!fromSock.isValid())
            return QVariant();
        return fromSock.data(ROLE_OBJID).toString();
    }
    case ROLE_OUTSOCK:
    {
        if (!toSock.isValid())
            return QVariant();
        return fromSock.data(ROLE_PARAM_NAME).toString();
    }
    case ROLE_INNODE:
    {
        if (!toSock.isValid())
            return QVariant();
        return toSock.data(ROLE_OBJID).toString();
    }
    case ROLE_INSOCK:
    {
        if (!toSock.isValid())
            return QVariant();
        return toSock.data(ROLE_PARAM_NAME).toString();
    }
    case ROLE_INSOCK_IDX:   return toSock;
    case ROLE_OUTSOCK_IDX:  return fromSock;
    case ROLE_INNODE_IDX:
    {
        if (!toSock.isValid())
            return QVariant();
        return toSock.data(ROLE_NODE_IDX);
    }
    case ROLE_OUTNODE_IDX:
    {
        if (!fromSock.isValid())
            return QVariant();
        return fromSock.data(ROLE_NODE_IDX);
    }
    default:
        return QVariant();
    }
    return QVariant();
}

bool LinkModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return false;   //do not support setting link directly.
}

void LinkModel::setInputSocket(const QModelIndex& index, const QModelIndex& sockIdx)
{
    int r = index.row();
    if (r < 0 || r >= m_items.size())
        return;

    m_items[r].toSock = sockIdx;
}

void LinkModel::setOutputSocket(const QModelIndex& index, const QModelIndex& sockIdx)
{
    int r = index.row();
    if (r < 0 || r >= m_items.size())
        return;

    m_items[r].fromSock = sockIdx;
}

QVariant LinkModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    return QVariant();
}

bool LinkModel::setHeaderData(int section, Qt::Orientation orientation, const QVariant& value, int role)
{
    return false;
}

QModelIndexList LinkModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return QModelIndexList();
}

bool LinkModel::removeRows(int row, int count, const QModelIndex& parent)
{
    if (row < 0 || row >= m_items.size())
        return false;

    beginRemoveRows(parent, row, row);
    m_items.remove(row);
    endRemoveRows();
    return true;
}

int LinkModel::addLink(const QModelIndex& fromSock, const QModelIndex& toSock)
{
    int row = m_items.size();
    beginInsertRows(QModelIndex(), row, row);

    _linkItem item;
    item.fromSock = fromSock;
    item.toSock = toSock;
    item.ident = UiHelper::generateUuid();
    m_items.append(item);

    endInsertRows();
    return row;
}

QModelIndex LinkModel::index(const QModelIndex& fromSock, const QModelIndex& toSock)
{
    for (int r = 0; r < m_items.size(); r++)
    {
        _linkItem& item = m_items[r];
        if (item.fromSock == fromSock && item.toSock == toSock)
            return index(r, 0);
    }
    return QModelIndex();
}

void LinkModel::clear()
{
    while (rowCount() > 0)
    {
        //safe to notify the remove msg.
        removeRows(0, 1);
    }
}
