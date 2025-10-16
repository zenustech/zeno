#pragma once

#ifndef __GEOMETRY_MODEL_H__
#define __GEOMETRY_MODEL_H__

#include <QObject>
#include <QAbstractTableModel>
#include <map>
#include <vector>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/UserData.h>
#include <QAbstractListModel>
#include <unordered_map>

namespace zeno {
    struct SceneTreeNode;
    struct SceneObject;
}

struct VertexInfo {
    int face;
    int point;
};

enum AttrColType {
    AttrInt,
    AttrFloat,
    AttrString,
};

struct AttributeInfo {
    std::string name;
    std::string showName;
    std::string _id;
    AttrColType type;
    char channel = 0;
};

class VertexModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    VertexModel(zeno::GeometryObject_Adapter* pObject, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

    void setGeoObject(zeno::GeometryObject_Adapter* spObject);

private:
    QMap<int, AttributeInfo> m_colMap;
    zeno::GeometryObject_Adapter* m_object;
    int m_nvertices;
};

class PointModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    PointModel(zeno::GeometryObject_Adapter* pObject, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

    void setGeoObject(zeno::GeometryObject_Adapter* pObject);

private:
    QMap<int, AttributeInfo> m_colMap;
    zeno::GeometryObject_Adapter* m_object;
    int m_npoints;
};

class FaceModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    FaceModel(zeno::GeometryObject_Adapter* pObject, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

    void setGeoObject(zeno::GeometryObject_Adapter* pObject);

private:
    QMap<int, AttributeInfo> m_colMap;
    zeno::GeometryObject_Adapter* m_object;

    int m_nfaces;
};

class GeomDetailModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    GeomDetailModel(zeno::GeometryObject_Adapter* pObject, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override { return 1; }
    int columnCount(const QModelIndex& parent) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    void setGeoObject(zeno::GeometryObject_Adapter* pObject);

private:
    QMap<int, AttributeInfo> m_colMap;
    zeno::GeometryObject_Adapter* m_object;
};

class GeomUserDataModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    GeomUserDataModel(zeno::GeometryObject_Adapter* pObject, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    void setGeoObject(zeno::GeometryObject_Adapter* pObject);

private:
    QVariant GeomUserDataModel::userDataToString(const zeno::zany& object) const;
    zeno::UserData* userData() const;

    zeno::GeometryObject_Adapter* m_object;
};

// SceneObject相关的通用模型类
class SceneObjectListModel : public QAbstractListModel
{
    Q_OBJECT
public:
    enum DataType {
        SceneTree,
        GeometryList,
        NodeToMatrix  // 添加NodeToMatrix类型
    };

    SceneObjectListModel(DataType type, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    void setSceneObject(zeno::SceneObject* pObject);
    std::string getKeyAt(int index) const;
    zeno::SceneTreeNode getTreeNodeAt(int index) const;

    zeno::SceneObject* getSceneObject();
private:
    DataType m_type;
    zeno::SceneObject* m_object;
    std::vector<std::string> m_keys;
};

class SceneObjectTableModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    enum DataType {
        NodeToId  // 只保留NodeToId类型
    };

    SceneObjectTableModel(DataType type, QObject* parent = nullptr);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    void setSceneObject(zeno::SceneObject* pObject);

private:
    DataType m_type;
    zeno::SceneObject* m_object;
    std::vector<std::string> m_keys;
};

#endif