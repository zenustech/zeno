#include "dictkeymodel.h"


DictKeyModel::DictKeyModel(QObject* parent)
    : QStandardItemModel(0, 2, parent)
{
    //add default key.
    //QStandardItem *pObjItem = new QStandardItem("");
    //appendRow({new QStandardItem("obj0"), pObjItem});
}

bool DictKeyModel::moveRows(const QModelIndex &sourceParent, int sourceRow, int count,
                  const QModelIndex &destinationParent, int destinationChild)
{
    QPersistentModelIndex sourceIdx = this->index(sourceRow, 0);
    const QString& key = sourceIdx.data().toString();
    //todo: get connect socket idx from index(sourceRow, 1).
    QStandardItem* pNewItem = new QStandardItem(key);
    //todo: pNewItem->setData(ROLE_OUTPUTSOCK,...)
    this->invisibleRootItem()->insertRow(destinationChild, pNewItem);
    //this->insertRows(destinationChild, 1);
    //this->setData(index(destinationChild, 0), key, Qt::DisplayRole);
    removeRows(sourceIdx.row(), 1);
    return true;
}