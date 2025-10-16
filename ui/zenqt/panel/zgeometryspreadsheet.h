#pragma once

#ifndef __ZGEOMETRY_SPREADSHEET_H__
#define __ZGEOMETRY_SPREADSHEET_H__

#include <QtWidgets>
#include <zeno/types/IGeometryObject.h>

namespace zeno {
    struct SceneTreeNode;
    struct SceneObject;
    struct ListObject;
    struct MaterialObject;
}
class ZToolBarButton;
class GraphModel;

class UnderlineItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit UnderlineItemDelegate(QObject* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

class ListObjItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit ListObjItemDelegate(QObject* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

private:
    UnderlineItemDelegate* m_underlineDelegate;
};

// BaseAttributeView组件声明
class BaseAttributeView : public QWidget
{
    Q_OBJECT
public:
    BaseAttributeView(QWidget* parent = nullptr);
    void setGeometryObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::GeometryObject_Adapter* object, QString nodeName);
    void clearModel();

public slots:
    void onNodeRemoved(QString nodename);
    void onNodeDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);

private:
    QLabel* m_lblNode;
    ZToolBarButton* m_vertex;
    ZToolBarButton* m_point;
    ZToolBarButton* m_face;
    ZToolBarButton* m_geom;
    ZToolBarButton* m_ud;

    GraphModel* m_model;
    QPersistentModelIndex m_nodeIdx;

    QStackedWidget* m_stackViews;
};

// 三个浮动widget类的声明
class SceneTreeNodeWidget : public QWidget
{
    Q_OBJECT
public:
    SceneTreeNodeWidget(QWidget* parent = nullptr);
    void setTreeNode(const zeno::SceneTreeNode& treeNode);
    void clearModel();

private:
    QLabel* m_matrixLabel;
    QLabel* m_visibilityLabel;
    QTableView* m_dataTableView;
    QStandardItemModel* m_dataModel;  // 添加model成员变量
};

class MatrixWidget : public QWidget
{
    Q_OBJECT
public:
    MatrixWidget(QWidget* parent = nullptr);
    void setMatrices(const std::vector<glm::mat4>& matrices);
    void clearModel();

private:
    QTableView* m_matrixTableView;
    QStandardItemModel* m_matrixModel;  // 添加model成员变量
};

// SceneObjView组件声明
class SceneObjView : public QWidget
{
    Q_OBJECT
public:
    SceneObjView(QWidget* parent = nullptr);
    void setSceneObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::SceneObject* pObject, QString nodename);
    void clearModel();

private slots:
    void onRadioButtonToggled(bool checked);  // 修改：改为单选按钮切换事件

    void onSceneTreeClicked(const QModelIndex& index);
    void onNodeToMatrixClicked(const QModelIndex& index);
    void onGeomListClicked(const QModelIndex& index);
    void onNodeRemoved(QString nodename);

private:
    QLabel* m_lblNode;
    QRadioButton* m_radioSceneTree;  // 修改：改为QRadioButton
    QRadioButton* m_radioNodeToMatrix;  // 修改：改为QRadioButton
    QRadioButton* m_radioNodeToId;  // 修改：改为QRadioButton
    QRadioButton* m_radioGeomList;  // 修改：改为QRadioButton
    QStackedWidget* m_stackViews;
    GraphModel* m_model;
    QPersistentModelIndex m_nodeIdx;

    // 保留组件指针，用于在模态对话框中显示
    SceneTreeNodeWidget* m_sceneTreeNodeWidget;
    MatrixWidget* m_matrixWidget;
    BaseAttributeView* m_baseAttributeView;  // 修改：直接使用BaseAttributeView替换GeometryWidget
};

// MaterialObjView组件声明
class MaterialObjView : public QWidget
{
    Q_OBJECT
public:
    MaterialObjView(QWidget* parent = nullptr);
    void setMaterialObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::MaterialObject* pObject, QString nodename);
    void clearModel();

private slots:
    void onNodeRemoved(QString nodename);

private:
    QLabel* m_lblNode;
    QTableView* m_materialTableView;
    QStandardItemModel* m_materialModel;
    GraphModel* m_model;
    QPersistentModelIndex m_nodeIdx;
};

// ListObjView组件声明
class ListObjView : public QWidget
{
    Q_OBJECT
public:
    ListObjView(QWidget* parent = nullptr);
    void setListObject(GraphModel* subgraph, QModelIndex nodeidx, zeno::ListObject* pObject);
    void clearModel();

private slots:
    void onExpandAllToggled(bool checked);
    void onItemClicked(const QModelIndex& index);
    void onNodeRemoved(QString nodename);

private:
    void addListObjectItems(QStandardItem* parentItem, zeno::ListObject* listObj, QVector<int> parentIndice, const QString& parentName = "");

    QLabel* m_lblNode;
    QCheckBox* m_expandCheckBox;
    QTreeView* m_listTreeView;
    QStandardItemModel* m_listModel;
    ListObjItemDelegate* m_underlineDelegate;
    GraphModel* m_model;
    QPersistentModelIndex m_nodeIdx;
    zeno::ListObject* m_currentListObject;

    BaseAttributeView* m_baseAttributeView;
    SceneObjView* m_sceneObjView;
    MaterialObjView* m_materialObjView;
};

class ZGeometrySpreadsheet : public QWidget
{
    Q_OBJECT
public:
    ZGeometrySpreadsheet(QWidget* parent = nullptr);
    void setGeometry(GraphModel* subgraph, QModelIndex nodeidx, std::unique_ptr<zeno::IObject>&& pObject);
    void clearModel();

private:
    std::unique_ptr<zeno::IObject> m_clone_obj;
    QStackedWidget* m_views;
};


#endif