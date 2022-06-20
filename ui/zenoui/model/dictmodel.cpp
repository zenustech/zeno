#include "dictmodel.h"


DictModel::DictModel(QObject *parent)
    : QStandardItemModel(parent)
{
}

//int DictModel::columnCount(const QModelIndex& parent) const
//{
//    return 3;
//}
//
//QVariant DictModel::headerData(int section, Qt::Orientation orientation, int role) const
//{
//    if ((section < 0) || ((orientation == Qt::Horizontal) && (section >= columnCount())) ||
//        ((orientation == Qt::Vertical) && (section >= rowCount()))) {
//        return QVariant();
//    }
//    if (orientation == Qt::Horizontal)
//    {
//        switch (section) {
//        case 0: return tr("Key");
//        case 1: return tr("Type");
//        case 2: return tr("Value");
//        }
//        return QVariant();
//    }
//    else
//    {
//        return section;
//    }
//}