#include "zenooutline.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/displaywidget.h"
#include <zenovis/ObjectsManager.h>

// 修改构造函数
OutlineItemModel::OutlineItemModel(QObject *parent)
    : QAbstractItemModel(parent)
    , rootItem(std::make_unique<OutlineItem>())  // 使用make_unique初始化
{
    setupModelData();
}

OutlineItemModel::~OutlineItemModel()
{
}

// 修改setupModelData
void OutlineItemModel::setupModelData()
{
    beginResetModel();
    
    rootItem = std::make_unique<OutlineItem>();  // 重置rootItem
    auto* meshItem = rootItem->addChild("Mesh");
    auto* matrixItem = rootItem->addChild("Matrixes");
    auto* sceneDescItem = rootItem->addChild("SceneDescriptor");
    auto* othersItem = rootItem->addChild("Others");

    if (auto main = zenoApp->getMainWindow()) {
        for (auto view : main->viewports()) {
            if (!view->isGLViewport()) {
                if (auto vis = view->getZenoVis()) {
                    if (auto sess = vis->getSession()) {
                        if (auto scene = sess->get_scene()) {
                            for (auto& [key, obj] : scene->objectsMan->pairsShared()) {
                                if (obj->userData().has("ResourceType")) {
                                    if (obj->userData().get2<std::string>("ResourceType", "none") == "Mesh") {
                                        meshItem->addChild(QString::fromStdString(key));
                                    } else if (obj->userData().get2<std::string>("ResourceType", "none") == "Matrixes") {
                                        matrixItem->addChild(QString::fromStdString(key));
                                    } else if (obj->userData().get2<std::string>("ResourceType", "none") == "SceneDescriptor") {
                                        sceneDescItem->addChild(QString::fromStdString(key));
                                    } else {
                                        othersItem->addChild(QString::fromStdString(key));
                                    }
                                }
                            }
                        }
                    }
                }
                rootItem->children[0]->name = QString::fromStdString("Mesh:" + std::to_string(meshItem->children.size()));
                rootItem->children[1]->name = QString::fromStdString("Matrixes:" + std::to_string(matrixItem->children.size()));
                rootItem->children[2]->name = QString::fromStdString("SceneDescriptor:" + std::to_string(sceneDescItem->children.size()));
                rootItem->children[3]->name = QString::fromStdString("Others:" + std::to_string(othersItem->children.size()));
                break;
            }
        }
    }
    
    endResetModel();
}

QModelIndex OutlineItemModel::index(int row, int column, const QModelIndex& parent) const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    OutlineItem* parentItem = parent.isValid() ? static_cast<OutlineItem*>(parent.internalPointer()) : rootItem.get();

    if (row >= parentItem->children.size() || row < 0)
        return QModelIndex();

    OutlineItem* childItem = parentItem->children[row].get();
    return childItem ? createIndex(row, column, childItem) : QModelIndex();
}

QModelIndex OutlineItemModel::parent(const QModelIndex &child) const
{
    if (!child.isValid())
        return QModelIndex();

    OutlineItem* childItem = static_cast<OutlineItem*>(child.internalPointer());
    OutlineItem* parentItem = childItem->parent;

    if (!parentItem || parentItem == rootItem.get())
        return QModelIndex();

    return createIndex(parentItem->row, 0, parentItem);  // 使用缓存的row
}

int OutlineItemModel::rowCount(const QModelIndex &parent) const
{
    OutlineItem *parentItem = parent.isValid() ? static_cast<OutlineItem*>(parent.internalPointer()) : rootItem.get();
    return parentItem->children.size();
}

int OutlineItemModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 1;
}

QVariant OutlineItemModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || role != Qt::DisplayRole)
        return QVariant();

    OutlineItem *item = static_cast<OutlineItem*>(index.internalPointer());
    return item->name;
}

// zenooutline实现
zenooutline::zenooutline(QWidget *parent)
    : QWidget(parent)
{
    setupTreeView();
}

zenooutline::~zenooutline()
{
}

void zenooutline::setupTreeView()
{
    m_treeView = new QTreeView(this);
    m_model = new OutlineItemModel(this);
    
    m_treeView->setModel(m_model);
    m_treeView->setHeaderHidden(true);
    m_treeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_treeView->setSelectionMode(QAbstractItemView::SingleSelection);
    
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(m_treeView);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);
    
    m_treeView->expandAll();

    if (auto main = zenoApp->getMainWindow()) {
        for (auto view : main->viewports()) {
            if (!view->isGLViewport()) {
                if (auto vis = view->getZenoVis()) {
                    connect(vis, &Zenovis::objectsUpdated, this, [=](int frame) {
                        m_model->setupModelData();
                        m_treeView->expandAll();
                    });
                    break;
                }
            }
        }
    }
}
