#include "zenooutline.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/displaywidget.h"
#include <zenovis/ObjectsManager.h>
#include <tinygltf/json.hpp>
#include "viewport/zenovis.h"
#include "viewport/displaywidget.h"
#include "viewport/zoptixviewport.h"

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

void OutlineItemModel::set_child_node(Json const&json, OutlineItemModel::OutlineItem *item, std::string name) {
    auto sub_node = item->addChild(QString::fromStdString(name));
    for (auto &value: json[name]["children"]) {
        set_child_node(json, sub_node, std::string(value));
    }
};
// 修改setupModelData
void OutlineItemModel::setupModelData()
{
    beginResetModel();
    
    rootItem = std::make_unique<OutlineItem>();  // 重置rootItem
    auto* staticSceneItem = rootItem->addChild("StaticScene");
    auto* dynamicSceneItem = rootItem->addChild("DynamicScene");
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
                                    } else if (ud.get2<std::string>("ResourceType", "none") == "SceneTree") {
                                        auto scene_tree = obj->userData().get2<std::string>("json");
                                        if (scene_tree == static_scene_tree_str) {
                                            continue;
                                        }
                                        Json json = Json::parse(scene_tree);
                                        if (json["type"] == "dynamic") {
                                            dynamic_scene_tree = json;
                                            dynamic_scene->from_json(json);
                                        }
                                        else if (json["type"] == "static") {
                                            static_scene_tree = json;
                                            static_scene_tree_str = scene_tree;
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
                if (!static_scene_tree.empty()) {
                    std::string root_name = static_scene_tree["root_name"];
                    set_child_node(static_scene_tree["scene_tree"], staticSceneItem, root_name);
                }
                if (!dynamic_scene_tree.empty()) {
                    std::string root_name = dynamic_scene_tree["root_name"];
                    set_child_node(dynamic_scene_tree["scene_tree"], dynamicSceneItem, root_name);
                }
                rootItem->children[2]->name = QString::fromStdString("Mesh:" + std::to_string(meshItem->children.size()));
                rootItem->children[3]->name = QString::fromStdString("Matrixes:" + std::to_string(matrixItem->children.size()));
                rootItem->children[5]->name = QString::fromStdString("Others:" + std::to_string(othersItem->children.size()));
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

bool zenooutline::eventFilter(QObject *watched, QEvent *event) {
    std::optional<glm::mat4> xform = {};
    if (watched == m_treeView) {
        auto *treeView = qobject_cast<QTreeView *>(watched);
        if (treeView) {
            if (event->type() == QEvent::KeyPress) {
                if (auto *keyEvent = dynamic_cast<QKeyEvent *>(event)) {
                    if(keyEvent->key() == Qt::Key_Up) {
                        xform = glm::translate(glm::mat4(1), glm::vec3(0, 1, 0));
                    }
                    else if(keyEvent->key() == Qt::Key_Down) {
                        xform = glm::translate(glm::mat4(1), glm::vec3(0, -1, 0));
                    }
                }
            }
        }
    }
    if (xform.has_value()) {
        if (this->cur_node.has_value()) {
            auto &[name, lmat, pmat] = cur_node.value();
            if (this->modified_xfroms.count(name) == 0) {
                this->modified_xfroms[name] = lmat;
            }
            auto g_mat = pmat * this->modified_xfroms[name];
            g_mat = xform.value() * g_mat;
            auto n_mat = glm::inverse(pmat) * g_mat;
            this->modified_xfroms[name] = n_mat;
            auto new_matrixs = m_model->dynamic_scene->to_flatten_structure_matrix(modified_xfroms);

            if(ZenoMainWindow *mainWin = zenoApp->getMainWindow()) {
                for(auto displaywidget : mainWin->viewports()) {
                    if(displaywidget) {
                        if(!displaywidget->isGLViewport()) {
                            ZOptixViewport* opixView = displaywidget->optixViewport();
                            emit opixView->sig_updateMatrix(new_matrixs);
                        }
                    }
                }
            }
        }
        return true;
    }
    else {
        return QObject::eventFilter(watched, event);
    }
}

void zenooutline::setupTreeView()
{
    m_treeView = new QTreeView(this);
    m_model = new OutlineItemModel(this);
    
    m_treeView->setModel(m_model);
    m_treeView->setHeaderHidden(true);
    m_treeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_treeView->setSelectionMode(QAbstractItemView::SingleSelection);
    m_treeView->installEventFilter(this);
    
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(m_treeView);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    connect(m_treeView, &QTreeView::clicked, this, [this](const QModelIndex &index) {
        if (index.isValid() == false) {
            return;
        }
        QVariant data = m_model->data(index, Qt::DisplayRole);
        auto object_name = data.toString().toStdString();

        std::vector<std::string> link;
        link.push_back(object_name);

        QModelIndex parentIndex = index.parent();
        while (parentIndex.isValid()) {
            QVariant parentData = m_model->data(parentIndex, Qt::DisplayRole);
            auto parent_name = parentData.toString().toStdString();
            link.push_back(parent_name);
            parentIndex = parentIndex.parent();
        }
        std::reverse(link.begin(), link.end());
        if (this->m_model != nullptr && link.size() >= 2) {
            Json *json = nullptr;
            if (link[0] == "StaticScene") {
                json = &this->m_model->static_scene_tree;
            }
            else {
                json = &this->m_model->dynamic_scene_tree;
            }
            Json &scene_tree = json->operator[]("scene_tree");
            Json &node_to_matrix = json->operator[]("node_to_matrix");
            glm::mat4 p_matrix = glm::mat4(1);
            glm::mat4 l_matrix = glm::mat4(1);
            for (auto idx = 1; idx < link.size(); idx++) {
                auto &node_name = link[idx];
                auto matrix_node_json = scene_tree[node_name];
                if (matrix_node_json.is_null()) {
                    break;
                }
                p_matrix = p_matrix * l_matrix;
                std::string matrix_name = matrix_node_json["matrix"];
                auto mat_json = node_to_matrix[matrix_name];
                for (auto i = 0; i < 4; i++) {
                    for (auto j = 0; j < 3; j++) {
                        int index = i * 3 + j;
                        l_matrix[i][j] = float(mat_json[index]);
                    }
                }
            }
            this->cur_node = {object_name, l_matrix, p_matrix};
        }

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
                m_treeView->scrollTo(index, QAbstractItemView::EnsureVisible);
                m_treeView->setCurrentIndex(index);
            }
        }
    });

    if (auto main = zenoApp->getMainWindow()) {
        for (auto view : main->viewports()) {
            if (!view->isGLViewport()) {
                if (auto vis = view->getZenoVis()) {
                    connect(vis, &Zenovis::objectsUpdated, this, [=](int frame) {
                        m_model->setupModelData();
                    });
                    break;
                }
            }
        }
    }
}
