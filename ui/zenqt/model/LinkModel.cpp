#include "LinkModel.h"
#include "../common.h"


LinkModel::LinkModel(QObject* parent)
    : _base(parent)
{

}

LinkModel::~LinkModel()
{

}

int LinkModel::rowCount(const QModelIndex& parent) const
{
    return m_items.length();
}

QVariant LinkModel::data(const QModelIndex& index, int role) const
{
    switch (role) {
        case ROLE_LINK_FROMPARAM_INFO:
        {
            const auto& info = m_items[index.row()];
            QModelIndex nodeIdx = info.fromParam.data(ROLE_NODEIDX).toModelIndex();
            const QString& nodeName = nodeIdx.data(ROLE_OBJID).toString();
            const QString& paramName = info.fromParam.data(ROLE_OBJNAME).toString();
            return QVariantList{ nodeName, paramName, false};
        }
        case ROLE_LINK_TOPARAM_INFO:
        {
            const auto& info = m_items[index.row()];
            QModelIndex nodeIdx = info.toParam.data(ROLE_NODEIDX).toModelIndex();
            const QString& nodeName = nodeIdx.data(ROLE_OBJID).toString();
            const QString& paramName = info.toParam.data(ROLE_OBJNAME).toString();
            return QVariantList{ nodeName, paramName, true };
        }
    }
    return QVariant();
}

QHash<int, QByteArray> LinkModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[ROLE_LINK_FROMPARAM_INFO] = "fromParam";
    roles[ROLE_LINK_TOPARAM_INFO] = "toParam";
    return roles;
}

bool LinkModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    m_items.removeAt(row);
    endRemoveRows();
    return true;
}

QModelIndex LinkModel::addLink(const QModelIndex& fromParam, const QModelIndex& toParam)
{
    int row = m_items.size();
    beginInsertRows(QModelIndex(), row, row);

    _linkItem item;
    item.fromParam = fromParam;
    item.toParam = toParam;
    m_items.append(item);

    endInsertRows();
    return index(row, 0);
}

