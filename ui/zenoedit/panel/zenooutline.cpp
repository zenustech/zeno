#include "zenooutline.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/displaywidget.h"
#include <zenovis/ObjectsManager.h>
#include <tinygltf/json.hpp>

using Json = nlohmann::json;

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
                std::vector<std::string> mesh_names;
                std::vector<std::string> matrix_names;
                auto sd_json = Json();
                if (auto vis = view->getZenoVis()) {
                    if (auto sess = vis->getSession()) {
                        if (auto scene = sess->get_scene()) {
                            for (auto& [key, obj] : scene->objectsMan->pairsShared()) {
                                auto &ud = obj->userData();
                                if (ud.has("ResourceType")) {
                                    auto object_name = ud.get2<std::string>("ObjectName", key);
                                    if (ud.get2<std::string>("ResourceType", "none") == "Mesh") {
                                        mesh_names.emplace_back(object_name);
                                    } else if (ud.get2<std::string>("ResourceType", "none") == "Matrixes") {
                                        matrix_names.emplace_back(object_name);
                                    } else if (ud.get2<std::string>("ResourceType", "none") == "SceneDescriptor") {
//                                        sceneDescItem->addChild(QString::fromStdString(object_name));
                                        if (obj->userData().has<std::string>("Scene")) {
                                            auto sd_str = obj->userData().get2<std::string>("Scene");
                                            sd_json = Json::parse(sd_str);
                                        }
                                    } else {
                                        othersItem->addChild(QString::fromStdString(object_name));
                                    }
                                }
                            }
                        }
                    }
                }
                std::sort(mesh_names.begin(), mesh_names.end());
                for (auto &mesh_name: mesh_names) {
                    meshItem->addChild(QString::fromStdString(mesh_name));
                }
                std::sort(matrix_names.begin(), matrix_names.end());
                for (auto &matrix_name: matrix_names) {
                    matrixItem->addChild(QString::fromStdString(matrix_name));
                }
                // SceneDescriptor
                if (sd_json.is_null() == false) {
                    if (sd_json.contains("BasicRenderInstances")) {
                        auto sub_root = sceneDescItem->addChild(QString::fromStdString("BasicRenderInstances"));
                        for (auto& [key, _value] : sd_json["BasicRenderInstances"].items()) {
                            sub_root->addChild(QString::fromStdString(key));
                        }
                    }

                    if (sd_json.contains("StaticRenderGroups")) {
                        auto sub_root = sceneDescItem->addChild(QString::fromStdString("StaticRenderGroups"));
                        auto *StaticRenderGroups = &sd_json["StaticRenderGroups"];
                        for (auto& [key, _value] : StaticRenderGroups->items()) {
                            auto sub_node = sub_root->addChild(QString::fromStdString(key));
                            for (auto& [key_0, _value_0] : StaticRenderGroups->operator[](key).items()) {
                                sub_node->addChild(QString::fromStdString(key_0));
                            }
                        }
                    }
                    if (sd_json.contains("DynamicRenderGroups")) {
                        auto sub_root = sceneDescItem->addChild(QString::fromStdString("DynamicRenderGroups"));
                        auto *DynamicRenderGroups = &sd_json["DynamicRenderGroups"];
                        for (auto& [key, _value] : DynamicRenderGroups->items()) {
                            auto sub_node = sub_root->addChild(QString::fromStdString(key));
                            for (auto& [key_0, _value_0] : DynamicRenderGroups->operator[](key).items()) {
                                sub_node->addChild(QString::fromStdString(key_0));
                            }
                        }
                    }
                }
                rootItem->children[0]->name = QString::fromStdString("Mesh:" + std::to_string(meshItem->children.size()));
                rootItem->children[1]->name = QString::fromStdString("Matrixes:" + std::to_string(matrixItem->children.size()));
                rootItem->children[2]->name = QString::fromStdString("SceneDescriptor");
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

    connect(m_treeView, &QTreeView::clicked, this, [this](const QModelIndex &index) {
        if (index.isValid() == false) {
            return;
        }
        QVariant data = m_model->data(index, Qt::DisplayRole);
        auto object_name = data.toString().toStdString();
        ZenoMainWindow *mainWin = zenoApp->getMainWindow();
        mainWin->onPrimitiveSelected({object_name});
    });

    connect(m_treeView, &QTreeView::doubleClicked, this, [this](const QModelIndex& index) {
        if (index.isValid() == false) {
            return;
        }
        QVariant data = m_model->data(index, Qt::DisplayRole);
        auto object_name = data.toString().toStdString();
        auto parent = index.parent();
        if (!parent.isValid()) {
            return;
        }
        auto grouproot = parent.parent();
        if (!grouproot.isValid()) {
            return;
        }
        int rowCount = m_model->rowCount(grouproot);
        for (int row = 0; row < rowCount; ++row) {
            QModelIndex index = m_model->index(row, 0, grouproot);
            if (index.data(Qt::DisplayRole).toString().toStdString() == object_name) {
                m_treeView->scrollTo(index, QAbstractItemView::PositionAtTop);
            }
        }
    });

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
