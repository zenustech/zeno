#include "dictkeymodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include "zassert.h"
#include "uihelper.h"
#include "command.h"


DictKeyModel::DictKeyModel(IGraphsModel* pGraphs, const QModelIndex& dictParam, QObject* parent)
    : QAbstractItemModel(parent)
    , m_dictParam(dictParam)
    , m_pGraphs(pGraphs)
    , m_bCollasped(false)
{
}

DictKeyModel::~DictKeyModel()
{
}

void DictKeyModel::clearAll()
{
    const QString& dictPath = m_dictParam.data(ROLE_OBJPATH).toString();
    while (rowCount() > 0)
    {
        m_pGraphs->addExecuteCommand(new DictKeyAddRemCommand(false, m_pGraphs, dictPath, 0));
    }
}

QModelIndex DictKeyModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount() || column < 0 || column >= columnCount())
        return QModelIndex();

    return createIndex(row, column, (quintptr)0);
}

QModelIndex DictKeyModel::index(const QString& keyName)
{
    for (int r = 0; r < rowCount(); r++)
    {
        if (m_items[r].key == keyName)
            return index(r, 0);
    }
    return QModelIndex();
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
    switch (role)
    {
        case Qt::DisplayRole:
        case ROLE_PARAM_NAME:
        {
            _DictItem &item = m_items[index.row()];
            if (index.column() == 0) {
                item.key = value.toString();
                emit dataChanged(index, index, QVector<int>{role});
                return true;
            }
            break;
        }
        case ROLE_ADDLINK:
        {
            _DictItem& item = m_items[index.row()];
            QPersistentModelIndex linkIdx = value.toPersistentModelIndex();
            ZASSERT_EXIT(linkIdx.isValid(), false);
            item.links.append(linkIdx);
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case ROLE_REMOVELINK:
        {
            QPersistentModelIndex linkIdx = value.toPersistentModelIndex();
            _DictItem& item = m_items[index.row()];
            item.links.removeAll(linkIdx);
            emit dataChanged(index, index, QVector<int>{role});
            return true;
        }
        case ROLE_COLLASPED:
        {
            m_bCollasped = value.toBool();
            break;
        }
    }
    return QAbstractItemModel::setData(index, value, role);
}

QVariant DictKeyModel::data(const QModelIndex& index, int role) const
{
    switch (role)
    {
    case Qt::DisplayRole:
    {
        const _DictItem &item = m_items[index.row()];
        if (index.column() == 0) {
            return item.key;
        }
        else if (index.column() == 1) {
            //todo: check input or output, then return by linkIdx.
            PARAM_CLASS cls = (PARAM_CLASS)this->data(index, ROLE_PARAM_CLASS).toInt();
            if (cls == PARAM_INNER_INPUT)
            {
                if (!item.links.isEmpty() && item.links[0].isValid())
                {
                    QModelIndex outNodeIdx = item.links[0].data(ROLE_OUTNODE_IDX).toModelIndex();
                    QString displayInfo = outNodeIdx.data(ROLE_OBJNAME).toString();
                    return displayInfo;
                }
            }
            else if (cls == PARAM_INNER_OUTPUT)
            {
                if (!item.links.isEmpty() && item.links[0].isValid())
                {
                    QModelIndex inNodeIdx = item.links[0].data(ROLE_INNODE_IDX).toModelIndex();
                    QString displayInfo = inNodeIdx.data(ROLE_OBJNAME).toString();
                    return displayInfo;
                }
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
            const _DictItem& item = m_items[index.row()];
            return QVariant::fromValue(item.links);
        }
    case ROLE_PARAM_NAME:
    case ROLE_VPARAM_NAME:
    {
        const _DictItem &item = m_items[index.row()];
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
        const _DictItem& item = m_items[index.row()];
        return item.links[0].data(role);
    }
    case ROLE_OBJID:
    case ROLE_NODE_IDX:
        return m_dictParam.data(role);
    case ROLE_PARAM_COREIDX:
        return m_dictParam;
    case ROLE_PARAM_CLASS:
    {
        PARAM_CLASS cls = (PARAM_CLASS)m_dictParam.data(role).toInt();
        if (cls == PARAM_INPUT)
            return PARAM_INNER_INPUT;
        else
            return PARAM_INNER_OUTPUT;
    }
    case ROLE_PARAM_TYPE:
        return "string";
    case ROLE_OBJPATH:
    {
        const _DictItem &item = m_items[index.row()];
        QString path;
        path = m_dictParam.data(ROLE_OBJPATH).toString() + "/" + item.key;
        return path;
    }
    case ROLE_COLLASPED:
    {
        return m_bCollasped;
    }
    }
    return QVariant();
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
    return true;
}

bool DictKeyModel::insertColumns(int column, int count, const QModelIndex& parent)
{
    return false;
}

bool DictKeyModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row + count - 1);
    //remove links first.
    for (auto linkIdx : m_items[row].links)
    {
        m_pGraphs->removeLink(linkIdx, true);
    }
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
    //only support simple move up/move down, so the actual movement is swaping the two elements.
    if (sourceParent != destinationParent || count != 1 || sourceRow == destinationChild)
        return false;

    bool bret = false;
    if (sourceRow < destinationChild)
        bret = beginMoveRows(sourceParent, sourceRow, sourceRow, destinationParent, destinationChild + 1);
    else
        bret = beginMoveRows(sourceParent, sourceRow, sourceRow, destinationParent, destinationChild);
    if (!bret)
        return bret;

    m_items.move(sourceRow, destinationChild);
    endMoveRows();
    return true;
}

bool DictKeyModel::moveColumns(const QModelIndex &sourceParent, int sourceColumn, int count,
                               const QModelIndex &destinationParent, int destinationChild)
{
    return false;
}

bool DictKeyModel::isCollasped() const
{
    return m_bCollasped;
}

void DictKeyModel::setCollasped(bool bCollasped)
{
    m_bCollasped = bCollasped;
}
