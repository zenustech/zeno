#include "geometrymodel.h"
#include <zeno/types/GeometryObject.h>
#include <zeno/extra/SceneAssembler.h>


void addCol(zeno::GeometryObject* pObject, zeno::GeoAttrGroup group, QMap<int, AttributeInfo>& colMapping, std::string name, int& nCol) {
    QString qName = QString::fromStdString(name);
    //TODO: pos这种要放到最前头
    std::string data_id, xattr_id, yattr_id, zattr_id, wattr_id;
#ifdef TRACE_GEOM_ATTR_DATA
    data_id = pObject->get_attr_data_id(group, name);
    xattr_id = pObject->get_attr_data_id(group, name, "x");
    yattr_id = pObject->get_attr_data_id(group, name, "y");
    zattr_id = pObject->get_attr_data_id(group, name, "z");
    wattr_id = pObject->get_attr_data_id(group, name, "w");
#endif

    zeno::GeoAttrType type = pObject->get_attr_type(group, name);
    if (type == zeno::ATTR_VEC2 || type == zeno::ATTR_VEC3 || type == zeno::ATTR_VEC4) {
        colMapping[nCol++] = AttributeInfo{ name, name + ".x", xattr_id, AttrFloat, 'x' };
        colMapping[nCol++] = AttributeInfo{ name, name + ".y", yattr_id, AttrFloat, 'y' };
        if (type == zeno::ATTR_VEC3 || type == zeno::ATTR_VEC4) {
            colMapping[nCol++] = AttributeInfo{ name, name + ".z", zattr_id, AttrFloat, 'z' };
            if (type == zeno::ATTR_VEC4) {
                colMapping[nCol++] = AttributeInfo{ name, name + ".w", wattr_id, AttrFloat, 'w' };
            }
        }
    }
    else {
        if (type == zeno::ATTR_FLOAT) {
            colMapping[nCol++] = AttributeInfo{ name, name, data_id, AttrFloat };
        }
        else if (type == zeno::ATTR_INT) {
            colMapping[nCol++] = AttributeInfo{ name, name, data_id, AttrInt };
        }
        else if (type == zeno::ATTR_STRING) {
            colMapping[nCol++] = AttributeInfo{ name, name, data_id, AttrString };
        }
    }
}

static QMap<int, AttributeInfo> initColMapping(zeno::GeometryObject* pObject, zeno::GeoAttrGroup group)
{
    QMap<int, AttributeInfo> colMapping;
    std::vector<std::string> names = pObject->get_attr_names(group);
    int nCol = 0;
    if (zeno::ATTR_VERTEX == group) {
        colMapping[nCol++] = AttributeInfo{ "Face-Vertex", "Face-Vertex", "", AttrInt};
        colMapping[nCol++] = AttributeInfo{ "Point Number", "Point Number", "", AttrInt};
    }

    for (auto name : names) {
        addCol(pObject, group, colMapping, name, nCol);
    }
    return colMapping;
}



/// <summary>
/// 
/// </summary>
/// <param name="pObject"></param>
VertexModel::VertexModel(zeno::GeometryObject_Adapter* spObject, QObject* parent) 
    : m_object(spObject)
    , m_nvertices(spObject->nvertices())
{
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_VERTEX);
    spObject->m_impl->register_add_vertex([this](int vertext_linearIdx) {
        if (vertext_linearIdx != -1) {
            beginInsertRows(QModelIndex(), vertext_linearIdx, vertext_linearIdx);
            m_nvertices++;
            endInsertRows();
        }
    });
    spObject->m_impl->register_remove_vertex([this](int vertext_linearIdx) {
        if (vertext_linearIdx != -1) {
            beginRemoveRows(QModelIndex(), vertext_linearIdx, vertext_linearIdx);
            m_nvertices--;
            endRemoveRows();
        }
    });
    spObject->m_impl->register_reset_vertices([&]() {
        beginResetModel();
        m_nvertices = spObject->nvertices();
        endResetModel();
    });
    //m_object->register_create_vertex_attr([this](std::string attrname) {
    //    beginInsertColumns(QModelIndex(), m_colMap.size(), m_colMap.size());
    //    addCol(m_object, zeno::ATTR_VERTEX, m_colMap, attrname, m_colMap.size());
    //    endInsertColumns();
    //});
    //m_object->register_delete_vertex_attr([this](std::string attrname) {
    //    int rmidx = -1;
    //    for (QMap<int, AttributeInfo>::const_iterator it = m_colMap.begin(); it != m_colMap.end(); ++it) {
    //        if (it.value().name == attrname) {
    //            rmidx = it.key();
    //            break;
    //        }
    //    }
    //    if (rmidx != -1) {
    //        beginRemoveColumns(QModelIndex(), rmidx, rmidx);
    //        m_colMap.remove(rmidx);
    //        QMap<int, AttributeInfo> newColmap;
    //        for (auto& it = m_colMap.begin(); it != m_colMap.end(); ++it) {
    //            if (it.key() > rmidx) {
    //                newColmap.insert(it.key() - 1, it.value());
    //            } else {
    //                newColmap.insert(it.key(), it.value());
    //            }
    //        }
    //        m_colMap.swap(newColmap);
    //        endRemoveColumns();
    //    }
    //});
}

QVariant VertexModel::data(const QModelIndex& index, int role) const {
    auto spObject = m_object;
    if (!spObject) {
        return QVariant();
    }
    int row = index.row(), col = index.column();
    if (role == Qt::DisplayRole) {
        if (col == 0 || col == 1) {
            //Point Number
            const auto& [face, vertidx, point] = spObject->m_impl->vertex_info(row);
            if (col == 0)
                return QString("%1:%2").arg(face).arg(vertidx);
            else
                return QString::number(point);
        }
        else {
            const AttributeInfo& info = m_colMap[col];
            if (info.type == AttrFloat) {
                float val = spObject->m_impl->get_elem<float>(zeno::ATTR_VERTEX, info.name, info.channel, row);
                return QString::number(val);
            }
            else if (info.type == AttrInt) {
                int val = spObject->m_impl->get_elem<int>(zeno::ATTR_VERTEX, info.name, info.channel, row);
                return QString::number(val);
            }
            else if (info.type == AttrString) {
                std::string val = spObject->m_impl->get_elem<std::string>(zeno::ATTR_VERTEX, info.name, info.channel, row);
                return QString::fromStdString(val);
            }
        }
    }
    return QVariant();
}

int VertexModel::rowCount(const QModelIndex& parent) const {
    return m_nvertices;
}

int VertexModel::columnCount(const QModelIndex& parent) const {
    return m_colMap.size();
}

QVariant VertexModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role == Qt::DisplayRole) {
        if (orientation == Qt::Horizontal) {
#ifdef TRACE_GEOM_ATTR_DATA
            return QString::fromStdString(m_colMap[section].showName + "(" + m_colMap[section]._id +")");
#else
            return QString::fromStdString(m_colMap[section].showName);
#endif
        }
        else {
            return QString::number(section);
        }
    }
    return QAbstractTableModel::headerData(section, orientation, role);
}

void VertexModel::setGeoObject(zeno::GeometryObject_Adapter* spObject)
{
    beginResetModel();
    m_object = spObject;
    m_nvertices = spObject->nvertices();
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_VERTEX);
    endResetModel();
}

bool VertexModel::removeRows(int row, int count, const QModelIndex& parent) {
    return QAbstractTableModel::removeRows(row, count, parent);
}

/// <summary>
///
/// </summary>
/// <param name="pObject"></param>
PointModel::PointModel(zeno::GeometryObject_Adapter* spObject, QObject* parent) : m_object(spObject), m_npoints(spObject->npoints()) {
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_POINT);
    spObject->m_impl->register_add_point([this](int ptnum) {
        if (ptnum != -1) {
            beginInsertRows(QModelIndex(), ptnum, ptnum);
            m_npoints++;
            endInsertRows();
        }
    });
    spObject->m_impl->register_remove_point([this](int ptnum) {
        if (ptnum != -1) {
            beginRemoveRows(QModelIndex(), ptnum, ptnum);
            m_npoints--;
            endRemoveRows();
        }
    });
    //m_object->register_create_point_attr([this](std::string attrname) {
    //    beginInsertColumns(QModelIndex(), m_colMap.size(), m_colMap.size());
    //    addCol(m_object, zeno::ATTR_POINT, m_colMap, attrname, m_colMap.size());
    //    endInsertColumns();
    //});
    //m_object->register_delete_point_attr([this](std::string attrname) {
    //    int rmidx = -1;
    //    for (QMap<int, AttributeInfo>::const_iterator it = m_colMap.begin(); it != m_colMap.end(); ++it) {
    //        if (it.value().name == attrname) {
    //            rmidx = it.key();
    //            break;
    //        }
    //    }
    //    if (rmidx != -1) {
    //        beginRemoveColumns(QModelIndex(), rmidx, rmidx);
    //        m_colMap.remove(rmidx);
    //        QMap<int, AttributeInfo> newColmap;
    //        for (auto& it = m_colMap.begin(); it != m_colMap.end(); ++it) {
    //            if (it.key() > rmidx) {
    //                newColmap.insert(it.key() - 1, it.value());
    //            }
    //            else {
    //                newColmap.insert(it.key(), it.value());
    //            }
    //        }
    //        m_colMap.swap(newColmap);
    //        endRemoveColumns();
    //    }
    //});
}

QVariant PointModel::data(const QModelIndex& index, int role) const {
    auto spObject = m_object;
    if (!spObject)
        return QVariant();
    int row = index.row(), col = index.column();
    if (role == Qt::DisplayRole) {
        const AttributeInfo& info = m_colMap[col];
        if (info.type == AttrFloat) {
            float val = spObject->m_impl->get_elem<float>(zeno::ATTR_POINT, info.name, info.channel, row);
            return QString::number(val);
        }
        else if (info.type == AttrInt) {
            int val = spObject->m_impl->get_elem<int>(zeno::ATTR_POINT, info.name, info.channel, row);
            return QString::number(val);
        }
        else if (info.type == AttrString) {
            std::string val = spObject->m_impl->get_elem<std::string>(zeno::ATTR_POINT, info.name, info.channel, row);
            return QString::fromStdString(val);
        }
    }
    return QVariant();
}

bool PointModel::setData(const QModelIndex& index, const QVariant& value, int role) {
    return false;
}

QVariant PointModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role == Qt::DisplayRole) {
        if (orientation == Qt::Horizontal) {
            const AttributeInfo& info = m_colMap[section];
#ifdef TRACE_GEOM_ATTR_DATA
            return QString::fromStdString(info.showName + "(" + info._id + ")");
#else
            return QString::fromStdString(info.showName);
#endif
        }
        else {
            return QString::number(section);
        }
    }
    return QAbstractTableModel::headerData(section, orientation, role);
}

void PointModel::setGeoObject(zeno::GeometryObject_Adapter* spObject)
{
    beginResetModel();
    m_object = spObject;
    m_npoints = spObject->npoints();
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_POINT);
    endResetModel();
}

int PointModel::rowCount(const QModelIndex& parent) const {
    return m_npoints;
}

int PointModel::columnCount(const QModelIndex& parent) const {
    return m_colMap.size();
}

bool PointModel::removeRows(int row, int count, const QModelIndex& parent) {
    return QAbstractTableModel::removeRows(row, count, parent);
}

/// <summary>
/// </summary>
/// <param name="pObject"></param>

FaceModel::FaceModel(zeno::GeometryObject_Adapter* spObject, QObject* parent) : m_object(spObject), m_nfaces(spObject->nfaces()) {
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_FACE);
    spObject->m_impl->register_add_face([this](int faceid) {
        if (faceid != -1) {
            beginInsertRows(QModelIndex(), faceid, faceid);
            m_nfaces++;
            endInsertRows();
        }
    });
    spObject->m_impl->register_remove_face([this](int faceid) {
        if (faceid != -1) {
            beginRemoveRows(QModelIndex(), faceid, faceid);
            m_nfaces--;
            endRemoveRows();
        }
    });
    spObject->m_impl->register_reset_faces([this]() {
        beginResetModel();
        auto _spObject = m_object;
        m_nfaces = _spObject->nfaces();
        endResetModel();
    });
    //m_object->register_create_face_attr([this](std::string attrname) {
    //    beginInsertColumns(QModelIndex(), m_colMap.size(), m_colMap.size());
    //    addCol(m_object, zeno::ATTR_FACE, m_colMap, attrname, m_colMap.size());
    //    endInsertColumns();
    //});
    //m_object->register_delete_face_attr([this](std::string attrname) {
    //    int rmidx = -1;
    //    for (QMap<int, AttributeInfo>::const_iterator it = m_colMap.begin(); it != m_colMap.end(); ++it) {
    //        if (it.value().name == attrname) {
    //            rmidx = it.key();
    //            break;
    //        }
    //    }
    //    if (rmidx != -1) {
    //        beginRemoveColumns(QModelIndex(), rmidx, rmidx);
    //        m_colMap.remove(rmidx);
    //        QMap<int, AttributeInfo> newColmap;
    //        for (auto& it = m_colMap.begin(); it != m_colMap.end(); ++it) {
    //            if (it.key() > rmidx) {
    //                newColmap.insert(it.key() - 1, it.value());
    //            }
    //            else {
    //                newColmap.insert(it.key(), it.value());
    //            }
    //        }
    //        m_colMap.swap(newColmap);
    //        endRemoveColumns();
    //    }
    //});
}

QVariant FaceModel::data(const QModelIndex& index, int role) const {
    auto spObject = m_object;
    if (!spObject) {
        return QVariant();
    }
    int row = index.row(), col = index.column();
    if (role == Qt::DisplayRole) {
        const AttributeInfo& info = m_colMap[col];
        if (info.type == AttrFloat) {
            float val = spObject->m_impl->get_elem<float>(zeno::ATTR_FACE, info.name, info.channel, row);
            return QString::number(val);
        }
        else if (info.type == AttrInt) {
            int val = spObject->m_impl->get_elem<int>(zeno::ATTR_FACE, info.name, info.channel, row);
            return QString::number(val);
        }
        else if (info.type == AttrString) {
            std::string val = spObject->m_impl->get_elem<std::string>(zeno::ATTR_FACE, info.name, info.channel, row);
            return QString::fromStdString(val);
        }
    }
    return QVariant();
}

int FaceModel::rowCount(const QModelIndex& parent) const {
    return m_nfaces;
}

int FaceModel::columnCount(const QModelIndex& parent) const {
    return m_colMap.size();
}

bool FaceModel::removeRows(int row, int count, const QModelIndex& parent) {
    return QAbstractTableModel::removeRows(row, count, parent);
}

QVariant FaceModel::headerData(int section, Qt::Orientation orientation, int role /*= Qt::DisplayRole*/) const
{
    if (role == Qt::DisplayRole) {
        if (orientation == Qt::Horizontal) {
            return QString::fromStdString(m_colMap[section].showName);
        }
        else {
            return QString::number(section);
        }
    }
    return QAbstractTableModel::headerData(section, orientation, role);
}

void FaceModel::setGeoObject(zeno::GeometryObject_Adapter* spObject)
{
    if (!spObject) {
        return;
    }
    beginResetModel();
    m_object = spObject;
    m_nfaces = spObject->nfaces();
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_FACE);
    endResetModel();
}


GeomDetailModel::GeomDetailModel(zeno::GeometryObject_Adapter* spObject, QObject* parent)
    : QAbstractTableModel(parent)
    , m_object(spObject)
{
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_GEO);
}

QVariant GeomDetailModel::data(const QModelIndex& index, int role) const
{
    auto spObject = m_object;
    if (!spObject)
        return QVariant();

    int row = index.row(), col = index.column();
    if (role == Qt::DisplayRole) {
        const AttributeInfo& info = m_colMap[col];
        if (info.type == AttrFloat) {
            float val = spObject->m_impl->get_elem<float>(zeno::ATTR_GEO, info.name, info.channel, row);
            return QString::number(val);
        }
        else if (info.type == AttrInt) {
            int val = spObject->m_impl->get_elem<int>(zeno::ATTR_GEO, info.name, info.channel, row);
            return QString::number(val);
        }
        else if (info.type == AttrString) {
            std::string val = spObject->m_impl->get_elem<std::string>(zeno::ATTR_GEO, info.name, info.channel, row);
            return QString::fromStdString(val);
        }
    }
    return QVariant();
}

int GeomDetailModel::columnCount(const QModelIndex& parent) const {
    return m_colMap.size();
}

QVariant GeomDetailModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role == Qt::DisplayRole) {
        if (orientation == Qt::Horizontal) {
            return QString::fromStdString(m_colMap[section].showName);
        }
        else {
            return QString::number(section);
        }
    }
    return QAbstractTableModel::headerData(section, orientation, role);
}

void GeomDetailModel::setGeoObject(zeno::GeometryObject_Adapter* spObject) {
    beginResetModel();
    m_object = spObject;
    m_colMap = initColMapping(spObject->m_impl.get(), zeno::ATTR_GEO);
    endResetModel();
}


GeomUserDataModel::GeomUserDataModel(zeno::GeometryObject_Adapter* pObject, QObject* parent)
    : QAbstractTableModel(parent)
    , m_object(pObject)
{
}

QVariant GeomUserDataModel::data(const QModelIndex& index, int role) const {
    auto pUserData = userData();
    if (!pUserData) return QVariant();

    if (role == Qt::DisplayRole) {
        auto it = std::next(pUserData->begin(), index.row());
        if (index.column() == 0) {
            return QString::fromStdString(it->first);
        } else if (index.column() == 1) {
            auto currentData = userDataToString(it->second);
            if (currentData.isValid()) {
                return currentData;
            } else {
                return QString("Invalid data");
            }
        }
    }
    return QVariant();
}

int GeomUserDataModel::rowCount(const QModelIndex& parent) const
{
    auto pUserData = userData();
    if (!pUserData) return 0;
    return pUserData->size();
}

int GeomUserDataModel::columnCount(const QModelIndex& parent) const {
    return 2;
}

zeno::UserData* GeomUserDataModel::userData() const {
    if (auto spObject = m_object) {
        return static_cast<zeno::UserData*>(spObject->userData());
    }
    else {
        return nullptr;
    }
}

QVariant GeomUserDataModel::userDataToString(const zeno::zany& object) const {
    if (zeno::objectIsLiterial<float>(object.get())) {
        auto v = zeno::objectToLiterial<float>(object);
        return QString::number(v);
    }
    else if (zeno::objectIsLiterial<int>(object.get())) {
        auto v = zeno::objectToLiterial<int>(object);
        return QString::number(v);
    }
    else if (zeno::objectIsLiterial<zeno::vec2f>(object.get())) {
        auto v = zeno::objectToLiterial<zeno::vec2f>(object);
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else if (zeno::objectIsLiterial<zeno::vec2i>(object.get())) {
        auto v = zeno::objectToLiterial<zeno::vec2i>(object);
        return QString("%1, %2").arg(v[0]).arg(v[1]);
    }
    else if (zeno::objectIsLiterial<zeno::vec3f>(object.get())) {
        auto v = zeno::objectToLiterial<zeno::vec3f>(object);
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else if (zeno::objectIsLiterial<zeno::vec3i>(object.get())) {
        auto v = zeno::objectToLiterial<zeno::vec3i>(object);
        return QString("%1, %2, %3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    else if (zeno::objectIsLiterial<zeno::vec4f>(object.get())) {
        auto v = zeno::objectToLiterial<zeno::vec4f>(object);
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else if (zeno::objectIsLiterial<zeno::vec4i>(object.get())) {
        auto v = zeno::objectToLiterial<zeno::vec4i>(object);
        return QString("%1, %2, %3, %4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    else if (zeno::objectIsLiterial<std::string>(object.get())) {
        auto v = zeno::objectToLiterial<std::string>(object);
        return QString(v.c_str());
    }
    return QVariant();
}

QVariant GeomUserDataModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role == Qt::DisplayRole) {
        if (orientation == Qt::Horizontal) {
            if (section == 0) {
                return "Key";
            } else if (section == 1) {
                return "Value";
            }
        } else {
            return QString::number(section);
        }
    }
    return QAbstractTableModel::headerData(section, orientation, role);
}

void GeomUserDataModel::setGeoObject(zeno::GeometryObject_Adapter* pObject) {
    beginResetModel();
    m_object = pObject;
    endResetModel();
}

// SceneObjectListModel实现
SceneObjectListModel::SceneObjectListModel(DataType type, QObject* parent)
    : QAbstractListModel(parent)
    , m_type(type)
{
}

QVariant SceneObjectListModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || !m_object) {
        return QVariant();
    }

    int row = index.row();

    if (role == Qt::DisplayRole) {
        if (row >= 0 && row < m_keys.size()) {
            QString key = QString::fromStdString(m_keys[row]);

            if (m_type == SceneTree) {
                // 显示场景树节点信息
                auto it = m_object->scene_tree.find(m_keys[row]);
                if (it != m_object->scene_tree.end()) {
                    const auto& node = it->second;
                    QString info = QString("%1: %2 (meshes: %3, children: %4)")
                        .arg(row)
                        .arg(key)
                        .arg(node.meshes.size())
                        .arg(node.children.size());
                    return info;
                }
            } else if (m_type == GeometryList) {
                // 显示几何对象信息
                auto it = m_object->geom_list.find(m_keys[row]);
                if (it != m_object->geom_list.end()) {
                    const auto& geom = it->second;
                    return QString("%1: %2").arg(row).arg(key);
                }
            } else if (m_type == NodeToMatrix) {
                // 显示节点名称和矩阵大小
                auto it = m_object->node_to_matrix.find(m_keys[row]);
                if (it != m_object->node_to_matrix.end()) {
                    const auto& matrices = it->second;
                    QString info = QString("%1: %2 (matrix: %3)")
                        .arg(row)
                        .arg(key)
                        .arg(matrices.size());
                    return info;
                }
            }
        }
    }

    return QVariant();
}

int SceneObjectListModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid() || !m_object) {
        return 0;
    }

    return m_keys.size();
}

QVariant SceneObjectListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
        if (m_type == SceneTree) {
            return "Scene Tree Nodes";
        } else if (m_type == GeometryList) {
            return "Geometry Objects";
        } else if (m_type == NodeToMatrix) {
            return "Node To Matrix Keys";
        }
    }
    return QVariant();
}

void SceneObjectListModel::setSceneObject(zeno::SceneObject* pObject)
{
    beginResetModel();
    m_object = pObject;
    m_keys.clear();

    if (pObject) {
        if (m_type == SceneTree) {
            for (const auto& pair : pObject->scene_tree) {
                m_keys.push_back(pair.first);
            }
        } else if (m_type == GeometryList) {
            for (const auto& pair : pObject->geom_list) {
                m_keys.push_back(pair.first);
            }
        } else if (m_type == NodeToMatrix) {
            for (const auto& pair : pObject->node_to_matrix) {
                m_keys.push_back(pair.first);
            }
        }
    }

    endResetModel();
}

std::string SceneObjectListModel::getKeyAt(int index) const
{
    if (index >= 0 && index < m_keys.size()) {
        return m_keys[index];
    }
    return "";
}

zeno::SceneTreeNode SceneObjectListModel::getTreeNodeAt(int index) const
{
    if (m_object && index >= 0 && index < m_keys.size()) {
        auto it = m_object->scene_tree.find(m_keys[index]);
        if (it != m_object->scene_tree.end()) {
            return it->second;
        }
    }
    return zeno::SceneTreeNode();
}

zeno::SceneObject* SceneObjectListModel::getSceneObject()
{
    return m_object;
}

// SceneObjectTableModel实现
SceneObjectTableModel::SceneObjectTableModel(DataType type, QObject* parent)
    : QAbstractTableModel(parent)
    , m_type(type)
{
}

QVariant SceneObjectTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || !m_object) {
        return QVariant();
    }

    int row = index.row();
    int col = index.column();

    if (role == Qt::DisplayRole) {
        if (row >= 0 && row < m_keys.size()) {
            QString key = QString::fromStdString(m_keys[row]);

            if (m_type == NodeToId) {
                auto it = m_object->node_to_id.find(m_keys[row]);
                if (it != m_object->node_to_id.end()) {
                    const auto& ids = it->second;
                    if (col == 0) {
                        // 第一列：节点名称
                        return key;
                    } else if (col == 1) {
                        // 第二列：将std::vector<int>拼接为string
                        QStringList idStrings;
                        for (int id : ids) {
                            idStrings.append(QString::number(id));
                        }
                        return idStrings.join(", ");
                    }
                }
            }
        }
    }

    return QVariant();
}

int SceneObjectTableModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid() || !m_object) {
        return 0;
    }

    return m_keys.size();
}

int SceneObjectTableModel::columnCount(const QModelIndex& parent) const
{
    if (parent.isValid()) {
        return 0;
    }

    return 2; // 节点名称、ID列表
}

QVariant SceneObjectTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
        switch (section) {
        case 0: return "Node Name";
        case 1: return "ID List";
        }
    }
    return QVariant();
}

void SceneObjectTableModel::setSceneObject(zeno::SceneObject* pObject)
{
    beginResetModel();
    m_object = pObject;
    m_keys.clear();

    if (pObject) {
        if (m_type == NodeToId) {
            for (const auto& pair : pObject->node_to_id) {
                m_keys.push_back(pair.first);
            }
        }
    }

    endResetModel();
}
