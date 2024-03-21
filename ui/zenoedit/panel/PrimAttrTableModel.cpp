//
// Created by zh on 2022/6/30.
//

#include "PrimAttrTableModel.h"
#include <zeno/types/PrimitiveObject.h>
#include "zeno/types/UserData.h"
#include "zeno/utils/format.h"
#include "zeno/utils/log.h"
#include <zeno/types/AttrVector.h>
#include <zeno/funcs/LiterialConverter.h>

const char* vecElementName[] = {"x", "y", "z", "w"};

using zeno::AttrAcceptAll;

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
        else if (sel_attr == "UVs") {
            return (int)(m_prim->uvs.size());
        }
        else {
            return (int)m_prim->userData().size();
        }
    }
    else {
        return 0;
    }
}

int PrimAttrTableModel::columnCount(const QModelIndex &parent) const {
    if (m_prim) {
        if (sel_attr == "Vertex") {
            return m_prim->verts.total_dim();
        }
        else if (sel_attr == "Tris") {
            return m_prim->tris.total_dim();
        }
        else if (sel_attr == "Points") {
            return m_prim->points.total_dim();
        }
        else if (sel_attr == "Lines") {
            return m_prim->lines.total_dim();
        }
        else if (sel_attr == "Quads") {
            return m_prim->quads.total_dim();
        }
        else if (sel_attr == "Polys") {
            return m_prim->polys.total_dim();
        }
        else if (sel_attr == "Loops") {
            return m_prim->loops.total_dim();
        }
        else if (sel_attr == "UVs") {
            return m_prim->uvs.total_dim();
        }
        else {
            return 1;
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
            return pointsData(index);
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
        else if (sel_attr == "UVs") {
            return uvsData(index);
        }
        else {
            auto it = std::next(m_prim->userData().begin(), index.row());
            
            auto currentData = userData(it->second);
            if (currentData.isValid()) {
                return currentData;
            }
        }
        return "-";
    }
    return QVariant();
}

template<typename T>
QString attrName(const zeno::AttrVector<T>& attr, zeno::AttrVectorIndex index) {
    if (index.attrIndex == 0) {
        return QString("pos");
    }

    return QString(attr.template attr_keys<AttrAcceptAll>()[index.attrIndex - 1].c_str());
}

QVariant PrimAttrTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (Qt::DisplayRole != role)
        return QVariant();

    if (orientation == Qt::Horizontal)
    {
        auto getHeaderName = [](const QString& name, const zeno::AttrVectorIndex& index) {
            if (index.attrDim <= 1) {
                return name;
            }
            return QString("%1[%2]").arg(name).arg(vecElementName[index.elementIndex]);
        };

        if (sel_attr == "Vertex") {
            auto index = m_prim->verts.attr_index(section);
            auto name = attrName(m_prim->verts, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "Tris") {
            auto index = m_prim->tris.attr_index(section);
            auto name = attrName(m_prim->tris, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "Points") {
            auto index = m_prim->points.attr_index(section);
            auto name = attrName(m_prim->points, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "Lines") {
            auto index = m_prim->lines.attr_index(section);
            auto name = attrName(m_prim->lines, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "Quads") {
            auto index = m_prim->quads.attr_index(section);
            auto name = attrName(m_prim->quads, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "Polys") {
            auto index = m_prim->polys.attr_index(section);
            auto name = attrName(m_prim->polys, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "Loops") {
            auto index = m_prim->loops.attr_index(section);
            auto name = attrName(m_prim->loops, index);
            return getHeaderName(name, index);
        }
        else if (sel_attr == "UVs") {
            auto index = m_prim->uvs.attr_index(section);
            auto name = attrName(m_prim->uvs, index);
            return getHeaderName(name, index);
        }
        else {
            return QString("Value");
        }
    }
    else if (orientation == Qt::Vertical)
    {
        if (sel_attr == "UserData") {
            auto it = std::next(m_prim->userData().begin(), section);
            return QString(it->first.c_str());
        }
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
void PrimAttrTableModel::setStrMapping(bool enable) {
    beginResetModel();
    enable_str_mapping = enable;
    endResetModel();
}

template<typename T>
QVariant attrData(const zeno::AttrVector<T> &attr, const QModelIndex& modelIndex, bool enable_str_mapping, zeno::UserData &ud) {
    int row = modelIndex.row();
    int column = modelIndex.column();

    auto index = attr.attr_index(column);
    if (index.attrIndex == 0) {
        if constexpr (zeno::is_vec_v<T>) {
            return attr.at(row)[index.elementIndex];
        }
        else {
            return attr.at(row);
        }
    }

    std::string attr_name = attr.template attr_keys<AttrAcceptAll>().at(index.attrIndex - 1);
    if (attr.template attr_is<float>(attr_name)) {
        return attr.template attr<float>(attr_name).at(row);
    }
    else if (attr.template attr_is<zeno::vec2f>(attr_name)) {
        auto v = attr.template attr<zeno::vec2f>(attr_name).at(row);
        return v[index.elementIndex];
    }
    else if (attr.template attr_is<zeno::vec3f>(attr_name)) {
        auto v = attr.template attr<zeno::vec3f>(attr_name).at(row);
        return v[index.elementIndex];
    }
    else if (attr.template attr_is<zeno::vec4f>(attr_name)) {
        auto v = attr.template attr<zeno::vec4f>(attr_name).at(row);
        return v[index.elementIndex];
    }
    else if (attr.template attr_is<int>(attr_name)) {
        if (enable_str_mapping && ud.get2<int>(attr_name+"_count", 0) > 0) {
            int v = attr.template attr<int>(attr_name).at(row);
            std::string name = ud.get2<std::string>(zeno::format("{}_{}", attr_name, v), std::string());
            return QString(name.c_str());
        }
        else {
            return attr.template attr<int>(attr_name).at(row);
        }
    }
    else if (attr.template attr_is<zeno::vec2i>(attr_name)) {
        auto v = attr.template attr<zeno::vec2i>(attr_name).at(row);
        return v[index.elementIndex];
    }
    else if (attr.template attr_is<zeno::vec3i>(attr_name)) {
        auto v = attr.template attr<zeno::vec3i>(attr_name).at(row);
        return v[index.elementIndex];
    }
    else if (attr.template attr_is<zeno::vec4i>(attr_name)) {
        auto v = attr.template attr<zeno::vec4i>(attr_name).at(row);
        if (enable_str_mapping && ud.get2<int>(attr_name+"_count", 0) > 0) {
            std::string name = ud.get2<std::string>(zeno::format("{}_{}", attr_name, v[index.elementIndex]), std::string());
            return QString(name.c_str());
        }
        else {
            return v[index.elementIndex];
        }
    }
    else {
        return QVariant();
    }
}

QVariant PrimAttrTableModel::vertexData(const QModelIndex &index) const {
    return attrData(m_prim->verts, index, enable_str_mapping, m_prim->userData());
}

QVariant PrimAttrTableModel::trisData(const QModelIndex &index) const {
    return attrData(m_prim->tris, index, enable_str_mapping, m_prim->userData());
}
QVariant PrimAttrTableModel::pointsData(const QModelIndex &index) const {
    return attrData(m_prim->points, index, enable_str_mapping, m_prim->userData());
}
QVariant PrimAttrTableModel::linesData(const QModelIndex &index) const {
    return attrData(m_prim->lines, index, enable_str_mapping, m_prim->userData());
}
QVariant PrimAttrTableModel::quadsData(const QModelIndex &index) const {
    return attrData(m_prim->quads, index, enable_str_mapping, m_prim->userData());
}
QVariant PrimAttrTableModel::polysData(const QModelIndex &index) const {
    return attrData(m_prim->polys, index, enable_str_mapping, m_prim->userData());
}
QVariant PrimAttrTableModel::loopsData(const QModelIndex &index) const {
    return attrData(m_prim->loops, index, enable_str_mapping, m_prim->userData());
}
QVariant PrimAttrTableModel::uvsData(const QModelIndex &index) const {
    return attrData(m_prim->uvs, index, enable_str_mapping, m_prim->userData());
}

QVariant PrimAttrTableModel::userData(const zeno::zany& object) const
{
    if (zeno::objectIsLiterial<float>(object)) {
        auto v = zeno::objectToLiterial<float>(object);
        return v;
    }
    else if (zeno::objectIsLiterial<int>(object)) {
        auto v = zeno::objectToLiterial<int>(object);
        return v;
    }
    else if (zeno::objectIsLiterial<zeno::vec2f>(object)) {
        auto v = zeno::objectToLiterial<zeno::vec2f>(object);
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else if (zeno::objectIsLiterial<zeno::vec2i>(object)) {
        auto v = zeno::objectToLiterial<zeno::vec2i>(object);
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else if (zeno::objectIsLiterial<zeno::vec3f>(object)) {
        auto v = zeno::objectToLiterial<zeno::vec3f>(object);
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else if (zeno::objectIsLiterial<zeno::vec3i>(object)) {
        auto v = zeno::objectToLiterial<zeno::vec3i>(object);
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else if (zeno::objectIsLiterial<zeno::vec4f>(object)) {
        auto v = zeno::objectToLiterial<zeno::vec4f>(object);
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else if (zeno::objectIsLiterial<zeno::vec4i>(object)) {
        auto v = zeno::objectToLiterial<zeno::vec4i>(object);
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else if (zeno::objectIsLiterial<std::string>(object)) {
        auto v = zeno::objectToLiterial<std::string>(object);
        return QString(v.c_str());
    }
    return QVariant();
}

zeno::zany PrimAttrTableModel::userDataByIndex(const QModelIndex& index) const
{
    auto it = std::next(m_prim->userData().begin(), index.row());
    if (it != m_prim->userData().end())
        return it->second;
    else
        return zeno::zany();
}
