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
        else if (sel_attr == "Tris") {
            return (int)(m_prim->tris.size());
        }
        else if (sel_attr == "Points") {
            return (int)(m_prim->points.size());
        }
        else if (sel_attr == "Lines") {
            return (int)(m_prim->lines.size());
        }
        else if (sel_attr == "Quads") {
            return (int)(m_prim->quads.size());
        }
        else if (sel_attr == "Polys") {
            return (int)(m_prim->polys.size());
        }
        else if (sel_attr == "Loops") {
            return (int)(m_prim->loops.size());
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
        else if (sel_attr == "Tris") {
            return 1 + (int)m_prim->tris.num_attrs();
        }
        else if (sel_attr == "Points") {
            return 1 + (int)m_prim->points.num_attrs();
        }
        else if (sel_attr == "Lines") {
            return 1 + (int)m_prim->lines.num_attrs();
        }
        else if (sel_attr == "Quads") {
            return 1 + (int)m_prim->quads.num_attrs();
        }
        else if (sel_attr == "Polys") {
            return 1 + (int)m_prim->polys.num_attrs();
        }
        else if (sel_attr == "Loops") {
            return 1 + (int)m_prim->loops.num_attrs();
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
            return vertexData(index);
        }
        else if (sel_attr == "Tris") {
            return trisData(index);
        }
        else if (sel_attr == "Points") {
            return trisData(index);
        }
        else if (sel_attr == "Lines") {
            return linesData(index);
        }
        else if (sel_attr == "Quads") {
            return quadsData(index);
        }
        else if (sel_attr == "Polys") {
            return polysData(index);
        }
        else if (sel_attr == "Loops") {
            return loopsData(index);
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
        else if (sel_attr == "Tris") {
            if (section == 0) {
                return QString("value");
            }
            else {
                return QString(m_prim->tris.attr_keys()[section - 1].c_str());
            }
        }
        else if (sel_attr == "Points") {
            if (section == 0) {
                return QString("value");
            }
            else {
                return QString(m_prim->points.attr_keys()[section - 1].c_str());
            }
        }
        else if (sel_attr == "Lines") {
            if (section == 0) {
                return QString("value");
            }
            else {
                return QString(m_prim->lines.attr_keys()[section - 1].c_str());
            }
        }
        else if (sel_attr == "Quads") {
            if (section == 0) {
                return QString("value");
            }
            else {
                return QString(m_prim->quads.attr_keys()[section - 1].c_str());
            }
        }
        else if (sel_attr == "Polys") {
            if (section == 0) {
                return QString("value");
            }
            else {
                return QString(m_prim->polys.attr_keys()[section - 1].c_str());
            }
        }
        else if (sel_attr == "Loops") {
            if (section == 0) {
                return QString("value");
            }
            else {
                return QString(m_prim->loops.attr_keys()[section - 1].c_str());
            }
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
QVariant PrimAttrTableModel::vertexData(const QModelIndex &index) const {
    std::string attr_name = m_prim->attr_keys()[index.column()];
    if (m_prim->attr_is<float>(attr_name)) {
        return m_prim->attr<float>(attr_name)[index.row()];
    }
    else if (m_prim->attr_is<zeno::vec2f>(attr_name)) {
        auto v = m_prim->attr<zeno::vec2f>(attr_name)[index.row()];
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else if (m_prim->attr_is<zeno::vec3f>(attr_name)) {
        auto v = m_prim->attr<zeno::vec3f>(attr_name)[index.row()];
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else if (m_prim->attr_is<zeno::vec4f>(attr_name)) {
        auto v = m_prim->attr<zeno::vec4f>(attr_name)[index.row()];
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else if (m_prim->attr_is<int>(attr_name)) {
        return m_prim->attr<int>(attr_name)[index.row()];
    }
    else if (m_prim->attr_is<zeno::vec2i>(attr_name)) {
        auto v = m_prim->attr<zeno::vec2i>(attr_name)[index.row()];
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else if (m_prim->attr_is<zeno::vec3i>(attr_name)) {
        auto v = m_prim->attr<zeno::vec3i>(attr_name)[index.row()];
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else if (m_prim->attr_is<zeno::vec4i>(attr_name)) {
        auto v = m_prim->attr<zeno::vec4i>(attr_name)[index.row()];
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    return QVariant();
}

QVariant PrimAttrTableModel::trisData(const QModelIndex &index) const {
    if (index.column() == 0) {
        auto v = m_prim->tris.at(index.row());
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else {
        std::string attr_name = m_prim->tris.attr_keys()[index.column() - 1];
        if (m_prim->tris.attr_is<float>(attr_name)) {
            return m_prim->tris.attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->tris.attr_is<zeno::vec2f>(attr_name)) {
            auto v = m_prim->tris.attr<zeno::vec2f>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->tris.attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->tris.attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->tris.attr_is<zeno::vec4f>(attr_name)) {
            auto v = m_prim->tris.attr<zeno::vec4f>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
        else if (m_prim->tris.attr_is<int>(attr_name)) {
            return m_prim->tris.attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->tris.attr_is<zeno::vec2i>(attr_name)) {
            auto v = m_prim->tris.attr<zeno::vec2i>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->tris.attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->tris.attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->tris.attr_is<zeno::vec4i>(attr_name)) {
            auto v = m_prim->tris.attr<zeno::vec4i>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
    }
    return QVariant();
}
QVariant PrimAttrTableModel::pointsData(const QModelIndex &index) const {
    if (index.column() == 0) {
        auto v = m_prim->points.at(index.row());
        return v;
    }
    else {
        std::string attr_name = m_prim->points.attr_keys()[index.column() - 1];
        if (m_prim->points.attr_is<float>(attr_name)) {
            return m_prim->points.attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->points.attr_is<zeno::vec2f>(attr_name)) {
            auto v = m_prim->points.attr<zeno::vec2f>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->points.attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->points.attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->points.attr_is<zeno::vec4f>(attr_name)) {
            auto v = m_prim->points.attr<zeno::vec4f>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
        else if (m_prim->points.attr_is<int>(attr_name)) {
            return m_prim->points.attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->points.attr_is<zeno::vec2i>(attr_name)) {
            auto v = m_prim->points.attr<zeno::vec2i>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->points.attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->points.attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->points.attr_is<zeno::vec4i>(attr_name)) {
            auto v = m_prim->points.attr<zeno::vec4i>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
    }
    return QVariant();
}
QVariant PrimAttrTableModel::linesData(const QModelIndex &index) const {
    if (index.column() == 0) {
        auto v = m_prim->lines.at(index.row());
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else {
        std::string attr_name = m_prim->lines.attr_keys()[index.column() - 1];
        if (m_prim->lines.attr_is<float>(attr_name)) {
            return m_prim->lines.attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->lines.attr_is<zeno::vec2f>(attr_name)) {
            auto v = m_prim->lines.attr<zeno::vec2f>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->lines.attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->lines.attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->lines.attr_is<zeno::vec4f>(attr_name)) {
            auto v = m_prim->lines.attr<zeno::vec4f>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
        else if (m_prim->lines.attr_is<int>(attr_name)) {
            return m_prim->lines.attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->lines.attr_is<zeno::vec2i>(attr_name)) {
            auto v = m_prim->lines.attr<zeno::vec2i>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->lines.attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->lines.attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->lines.attr_is<zeno::vec4i>(attr_name)) {
            auto v = m_prim->lines.attr<zeno::vec4i>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
    }
    return QVariant();
}
QVariant PrimAttrTableModel::quadsData(const QModelIndex &index) const {
    if (index.column() == 0) {
        auto v = m_prim->quads.at(index.row());
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else {
        std::string attr_name = m_prim->quads.attr_keys()[index.column() - 1];
        if (m_prim->quads.attr_is<float>(attr_name)) {
            return m_prim->quads.attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->quads.attr_is<zeno::vec2f>(attr_name)) {
            auto v = m_prim->quads.attr<zeno::vec2f>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->quads.attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->quads.attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->quads.attr_is<zeno::vec4f>(attr_name)) {
            auto v = m_prim->quads.attr<zeno::vec4f>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
        else if (m_prim->quads.attr_is<int>(attr_name)) {
            return m_prim->quads.attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->quads.attr_is<zeno::vec2i>(attr_name)) {
            auto v = m_prim->quads.attr<zeno::vec2i>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->quads.attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->quads.attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->quads.attr_is<zeno::vec4i>(attr_name)) {
            auto v = m_prim->quads.attr<zeno::vec4i>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
    }
    return QVariant();
}
QVariant PrimAttrTableModel::polysData(const QModelIndex &index) const {
    if (index.column() == 0) {
        auto v = m_prim->polys.at(index.row());
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else {
        std::string attr_name = m_prim->polys.attr_keys()[index.column() - 1];
        if (m_prim->polys.attr_is<float>(attr_name)) {
            return m_prim->polys.attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->polys.attr_is<zeno::vec2f>(attr_name)) {
            auto v = m_prim->polys.attr<zeno::vec2f>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->polys.attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->polys.attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->polys.attr_is<zeno::vec4f>(attr_name)) {
            auto v = m_prim->polys.attr<zeno::vec4f>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
        else if (m_prim->polys.attr_is<int>(attr_name)) {
            return m_prim->polys.attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->polys.attr_is<zeno::vec2i>(attr_name)) {
            auto v = m_prim->polys.attr<zeno::vec2i>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->polys.attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->polys.attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->polys.attr_is<zeno::vec4i>(attr_name)) {
            auto v = m_prim->polys.attr<zeno::vec4i>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
    }
    return QVariant();
}
QVariant PrimAttrTableModel::loopsData(const QModelIndex &index) const {
    if (index.column() == 0) {
        auto v = m_prim->loops.at(index.row());
        return v;
    }
    else {
        std::string attr_name = m_prim->loops.attr_keys()[index.column() - 1];
        if (m_prim->loops.attr_is<float>(attr_name)) {
            return m_prim->loops.attr<float>(attr_name)[index.row()];
        }
        else if (m_prim->loops.attr_is<zeno::vec2f>(attr_name)) {
            auto v = m_prim->loops.attr<zeno::vec2f>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->loops.attr_is<zeno::vec3f>(attr_name)) {
            auto v = m_prim->loops.attr<zeno::vec3f>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->loops.attr_is<zeno::vec4f>(attr_name)) {
            auto v = m_prim->loops.attr<zeno::vec4f>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
        else if (m_prim->loops.attr_is<int>(attr_name)) {
            return m_prim->loops.attr<int>(attr_name)[index.row()];
        }
        else if (m_prim->loops.attr_is<zeno::vec2i>(attr_name)) {
            auto v = m_prim->loops.attr<zeno::vec2i>(attr_name)[index.row()];
            return QString("%1, %2").arg(v[0]).arg(v[1]);
        }
        else if (m_prim->loops.attr_is<zeno::vec3i>(attr_name)) {
            auto v = m_prim->loops.attr<zeno::vec3i>(attr_name)[index.row()];
            return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
        }
        else if (m_prim->loops.attr_is<zeno::vec4i>(attr_name)) {
            auto v = m_prim->loops.attr<zeno::vec4i>(attr_name)[index.row()];
            return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
        }
    }
    return QVariant();
}
