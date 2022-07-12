//
// Created by zh on 2022/6/30.
//

#include "PrimAttrTableModel.h"
#include <zeno/types/PrimitiveObject.h>
#include "zeno/utils/logger.h"

PrimAttrTableModel::PrimAttrTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int PrimAttrTableModel::rowCount(const QModelIndex &parent) const {
    if (m_prim) {
        return (int)(m_prim->verts.size());
    }
    else {
        return 0;
    }
}

int PrimAttrTableModel::columnCount(const QModelIndex &parent) const {
    if (m_prim) {
        return (int)m_prim->num_attrs();
    }
    else {
        return 0;
    }
}

QVariant PrimAttrTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (Qt::TextAlignmentRole == role)
    {
        return int(Qt::AlignLeft | Qt::AlignVCenter);
    }
    else if (Qt::DisplayRole == role)
    {
        std::string attr_name = m_prim->attr_keys()[index.column()];
        if (m_prim->attr_is<float>(attr_name)) {
            return m_prim->attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->attr_is<int>(attr_name)) {
            return m_prim->attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        return "-";
    }
    return QVariant();
}

QVariant PrimAttrTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (Qt::DisplayRole != role)
        return QVariant();

    if (orientation == Qt::Horizontal)
    {
        return QString(m_prim->attr_keys()[section].c_str());
    }
    else if (orientation == Qt::Vertical)
    {
        return section;
    }
    return QVariant();
}


void PrimAttrTableModel::setModelData(zeno::PrimitiveObject *prim) {
    beginResetModel();
    m_prim = prim;
    endResetModel();
}
