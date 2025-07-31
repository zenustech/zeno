#include "zenooutline.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/displaywidget.h"
#include <zenovis/ObjectsManager.h>
#include <tinygltf/json.hpp>
#include "viewport/zenovis.h"
#include "viewport/displaywidget.h"
#include "viewport/zoptixviewport.h"
#include "zeno/utils/string.h"

using Json = nlohmann::json;

// 修改构造函数
OutlineItemModel::OutlineItemModel(QObject *parent)
    : QAbstractItemModel(parent)
    , rootItem(std::make_unique<OutlineItem>())  // 使用make_unique初始化
{
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
void OutlineItemModel::setupModelDataFromMessage(Json const& content)
{
    beginResetModel();

    rootItem = std::make_unique<OutlineItem>();  // 重置rootItem
    auto* staticSceneItem = rootItem->addChild("StaticScene");
    auto* dynamicSceneItem = rootItem->addChild("DynamicScene");

    if (content.contains("StaticSceneTree")) {
        std::string root_name = content["StaticSceneTree"]["root_name"];
        set_child_node(content["StaticSceneTree"]["scene_tree"], staticSceneItem, root_name);
    }
    if (content.contains("DynamicSceneTree")) {
        std::string root_name = content["DynamicSceneTree"]["root_name"];
        set_child_node(content["DynamicSceneTree"]["scene_tree"], dynamicSceneItem, root_name);
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

    if (auto main = zenoApp->getMainWindow()) {
        for (DisplayWidget* view : main->viewports()) {
            if (auto optxview = view->optixViewport()) {
                connect(optxview, &ZOptixViewport::sig_viewportSendToOutline, this, [this](QString content) {
                    Json msg = Json::parse(content.toStdString());
                    if (msg["MessageType"] == "SceneTree" && this->m_model) {
                        this->m_model->setupModelDataFromMessage(msg);
                    }
                });
            }
        }
    }
    Json msg;
    msg["MessageType"] = "Init";
    sendOptixMessage(msg);
}

zenooutline::~zenooutline()
{
}

bool zenooutline::eventFilter(QObject *watched, QEvent *event) {
    std::string mode;
    std::string axis;
    bool local_space;
    if (auto main = zenoApp->getMainWindow()) {
        for (DisplayWidget* view : main->viewports()) {
            if (ZOptixViewport* optxview = view->optixViewport()) {
                std::tie(mode, axis, local_space) = optxview->get_srt_mode_axis();
            }
        }
    }
    bool changed = false;
    if (watched == m_treeView) {
        auto *treeView = qobject_cast<QTreeView *>(watched);
        if (treeView) {
            if (event->type() == QEvent::KeyPress) {
                if (auto *keyEvent = dynamic_cast<QKeyEvent *>(event)) {
                    if (keyEvent->key() == Qt::Key_R) {
                        mode = "Rotate";
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_E) {
                        mode = "Scale";
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_T) {
                        mode = "Translate";
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_X) {
                        if (zeno::str_contains(axis, "X")) {
                            axis = zeno::remove_all(axis, "X");
                        }
                        else {
                            axis += 'X';
                        }
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_Y) {
                        if (zeno::str_contains(axis, "Y")) {
                            axis = zeno::remove_all(axis, "Y");
                        }
                        else {
                            axis += 'Y';
                        }
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_Z) {
                        if (zeno::str_contains(axis, "Z")) {
                            axis = zeno::remove_all(axis, "Z");
                        }
                        else {
                            axis += 'Z';
                        }
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_L) {
                        local_space = !local_space;
                        changed = true;
                    }
                    else if (keyEvent->key() == Qt::Key_Escape) {
                        mode = "";
                        axis = "";
                        local_space = false;
                        changed = true;
                    }
                    if (axis.size() >= 2) {
                        std::sort(axis.begin(), axis.end());
                    }
                }
            }
        }
    }
    if (changed) {
        if (auto main = zenoApp->getMainWindow()) {
            for (DisplayWidget* view : main->viewports()) {
                if (ZOptixViewport* optxview = view->optixViewport()) {
                    optxview->set_srt_mode_axis(mode, axis, local_space);
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
        if (link.size() >= 2) {
            std::reverse(link.begin(), link.end());
            if (link[0] == "StaticScene" || link[0] == "DynamicScene") {
                Json msg;
                msg["MessageType"] = "Select";
                msg["Content"] = link;

                sendOptixMessage(msg);
            }
        }

        ZenoMainWindow *mainWin = zenoApp->getMainWindow();
        mainWin->onPrimitiveSelected({object_name});
    });
}

void zenooutline::sendOptixMessage(Json &msg) {
    if (auto main = zenoApp->getMainWindow()) {
        for (DisplayWidget* view : main->viewports()) {
            ZOptixViewport* optxview = view->optixViewport();
            QString msg_str = QString::fromStdString(msg.dump());
            emit optxview->sig_sendOptixMessage(msg_str);
        }
    }

}
