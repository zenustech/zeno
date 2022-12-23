#include "dictkeymodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include "zassert.h"
#include "uihelper.h"


DictKeyModel::DictKeyModel(IGraphsModel* pGraphs, const QModelIndex& dictParam, QObject* parent)
    : QAbstractItemModel(parent)
    , m_dictParam(dictParam)
    , m_pGraphs(pGraphs)
{
}

QModelIndex DictKeyModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount() || column < 0 || column >= columnCount())
        return QModelIndex();

    return createIndex(row, column, (quintptr)0);
}

QModelIndex DictKeyModel::parent(const QModelIndex& child) const
{
    return QModelIndex();
}

int DictKeyModel::rowCount(const QModelIndex& parent) const
{
    return m_items.size();
}

int DictKeyModel::columnCount(const QModelIndex& parent) const
{
    return 2;
}

bool DictKeyModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    _DictItem& item = m_items[index.row()];
    switch (role)
    {
        case Qt::DisplayRole:
        {
            if (index.column() == 0) {
                item.key = value.toString();
                emit dataChanged(index, index, QVector<int>{role});
                return true;
            }
            break;
        }
        case ROLE_ADDLINK:
        {
            QPersistentModelIndex linkIdx = value.toPersistentModelIndex();
            ZASSERT_EXIT(linkIdx.isValid(), false);
            item.link = linkIdx;
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case ROLE_REMOVELINK:
        {
            item.link = QModelIndex();
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
    }
    return QAbstractItemModel::setData(index, value, role);
}

QVariant DictKeyModel::data(const QModelIndex& index, int role) const
{
    const _DictItem& item = m_items[index.row()];
    switch (role)
    {
    case Qt::DisplayRole:
    {
        if (index.column() == 0) {
            return item.key;
        }
        else if (index.column() == 1) {
            //todo: check input or output, then return by linkIdx.
            PARAM_CLASS cls = (PARAM_CLASS)this->data(index, ROLE_PARAM_SOCKETTYPE).toInt();
            if (cls == PARAM_INNER_INPUT) {
                QModelIndex outNodeIdx = item.link.data(ROLE_OUTNODE_IDX).toModelIndex();
                QString displayInfo = outNodeIdx.data(ROLE_OBJNAME).toString();
                return displayInfo;
            }
            else if (cls == PARAM_INNER_OUTPUT) {
                QModelIndex inNodeIdx = item.link.data(ROLE_INNODE_IDX).toModelIndex();
                QString displayInfo = inNodeIdx.data(ROLE_OBJNAME).toString();
                return displayInfo;
            }
            return "";
        }
        break;
    }
    case ROLE_VPARAM_IS_COREPARAM:
        return true;
    //mock param role
    case ROLE_PARAM_CTRL:       return CONTROL_NONE;
    case ROLE_PARAM_SOCKPROP:   return SOCKPROP_EDITABLE;
    case ROLE_PARAM_LINKS:
        {
            PARAM_LINKS links;
            QModelIndex linkIdx = item.link;
            if (linkIdx.isValid())
                links.append(linkIdx);
            return QVariant::fromValue(links);
        }
    case ROLE_PARAM_NAME:
    case ROLE_VPARAM_NAME:
    {
        if (index.column() == 0) {
            return item.key;
        }
        return "";
    }
    case ROLE_OUTNODE:
    case ROLE_OUTSOCK:
    case ROLE_OUTNODE_IDX:
    case ROLE_OUTSOCK_IDX:
    {
        //todo: output case
        QModelIndex linkIdx = data(index, ROLE_LINK_IDX).toModelIndex();
        return linkIdx.data(role);
    }
    case ROLE_LINK_IDX:
    {
        QModelIndex linkIdx = item.link;
        return linkIdx;
    }
    case ROLE_OBJID:
    case ROLE_NODE_IDX:
        return m_dictParam.data(role);
    case ROLE_PARAM_SOCKETTYPE:
    {
        PARAM_CLASS cls = (PARAM_CLASS)m_dictParam.data(role).toInt();
        if (cls == PARAM_INPUT)
            return PARAM_INNER_INPUT;
        else
            return PARAM_INNER_OUTPUT;
    }
    case ROLE_PARAM_TYPE:
        return "string";
    default:
        return QVariant();
    }
}

bool DictKeyModel::insertRows(int row, int count, const QModelIndex& parent)
{
    if (count > 1)
        return false;
    beginInsertRows(parent, row, row + count - 1);
    _DictItem item;
    //we can init will a new key name.
    QStringList keys;
    for (int r = 0; r < rowCount(); r++) {
        const QModelIndex &idxKey = index(r, 0);
        keys.append(idxKey.data().toString());
    }
    const QString& newKeyName = UiHelper::getUniqueName(keys, "obj", false);
    item.key = newKeyName;
    m_items.insert(row, item);
    endInsertRows();
}

bool DictKeyModel::insertColumns(int column, int count, const QModelIndex& parent)
{
    return false;
}

bool DictKeyModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row + count - 1);
    //remove link first.
    QPersistentModelIndex linkIdx = m_items[row].link;
    m_pGraphs->removeLink(linkIdx);
    m_items.removeAt(row);
    endRemoveRows();
    return true;
}

bool DictKeyModel::removeColumns(int column, int count, const QModelIndex& parent)
{
    return false;
}

bool DictKeyModel::moveRows(const QModelIndex &sourceParent, int sourceRow, int count,
                  const QModelIndex &destinationParent, int destinationChild)
{
    beginMoveRows(sourceParent, sourceRow, sourceRow, destinationParent, destinationChild);
    _DictItem dstItem = m_items[destinationChild];
    m_items[destinationChild] = m_items[sourceRow];
    m_items[sourceRow] = dstItem;
    endMoveRows();
    return true;
}

bool DictKeyModel::moveColumns(const QModelIndex &sourceParent, int sourceColumn, int count,
                               const QModelIndex &destinationParent, int destinationChild)
{
    return false;
}
