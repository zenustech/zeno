//
// Created by zh on 2022/6/30.
//

#include "PrimAttrTableModel.h"
#include <zeno/types/PrimitiveObject.h>
#include "zeno/types/UserData.h"
#include <zeno/funcs/LiterialConverter.h>

PrimAttrTableModel::PrimAttrTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int PrimAttrTableModel::rowCount(const QModelIndex &parent) const {
    if (m_prim) {
        if (sel_attr == "Vertex") {
            return (int)(m_prim->verts.size());
        }
        else {
            return 1;
        }
    }
    else {
        return 0;
    }
}

int PrimAttrTableModel::columnCount(const QModelIndex &parent) const {
    if (m_prim) {
        if (sel_attr == "Vertex") {
            return (int)m_prim->num_attrs();
        }
        else {
            return m_prim->userData().size();
        }
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
        if (sel_attr == "Vertex") {
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
        }
        else {
            auto it = m_prim->userData().begin();
            auto i = index.column();
            while (i) {
                it++;
                i--;
            }
            if (zeno::objectIsLiterial<float>(it->second)) {
                auto v = zeno::objectToLiterial<float>(it->second);
                return v;
            }
            else if (zeno::objectIsLiterial<int>(it->second)) {
                auto v = zeno::objectToLiterial<int>(it->second);
                return v;
            }
            else if (zeno::objectIsLiterial<zeno::vec3f>(it->second)) {
                auto v = zeno::objectToLiterial<zeno::vec3f>(it->second);
                return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
            }
            else if (zeno::objectIsLiterial<zeno::vec3i>(it->second)) {
                auto v = zeno::objectToLiterial<zeno::vec3i>(it->second);
                return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
            }
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
        if (sel_attr == "Vertex") {
            return QString(m_prim->attr_keys()[section].c_str());
        }
        else {
            auto it = m_prim->userData().begin();
            auto i = section;
            while (i) {
                it++;
                i--;
            }
            return QString(it->first.c_str());
        }
    }
    else if (orientation == Qt::Vertical)
    {
        return section;
    }
    return QVariant();
}


void PrimAttrTableModel::setModelData(zeno::PrimitiveObject *prim) {
    beginResetModel();
    if (prim)
        m_prim = std::make_shared<zeno::PrimitiveObject>(*prim);
    else
        m_prim = nullptr;
    endResetModel();
}

void PrimAttrTableModel::setSelAttr(std::string sel_attr_) {
    beginResetModel();
    sel_attr = sel_attr_;
    endResetModel();
}
