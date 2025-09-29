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
    std::vector<std::string> children;
    for (auto &value: json[name]["children"]) {
        children.emplace_back(std::string(value));
    }
    std::sort(children.begin(), children.end());
    for (auto &value: children) {
        set_child_node(json, sub_node, std::string(value));
    }
};
// 修改setupModelData
void OutlineItemModel::setupModelDataFromMessage(Json const& content)
{
    beginResetModel();

    rootItem = std::make_unique<OutlineItem>();  // 重置rootItem
    auto* lights = rootItem->addChild("Lights");
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
    if (content.contains("Lights")) {
        std::vector<std::string> children;
        for (auto &value: content["Lights"]) {
            children.emplace_back(std::string(value));
        }
        std::sort(children.begin(), children.end());
        for (auto &value: children) {
            lights->addChild(QString::fromStdString(value));
        }
    }

    endResetModel();
}

void OutlineItemModel::clearModelData()
{
    beginResetModel();

    rootItem = std::make_unique<OutlineItem>();

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
                    if (this->m_model) {
                        if (msg["MessageType"] == "SceneTree") {
                            this->m_model->setupModelDataFromMessage(msg);
                        }
                        else if (msg["MessageType"] == "CleanupAssets") {
                            this->m_model->clearModelData();
                        }
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
    if (auto main = zenoApp->getMainWindow()) {
        for (DisplayWidget* view : main->viewports()) {
            if (ZOptixViewport* optxview = view->optixViewport()) {
            }
        }
    }
    bool changed = false;
    if (watched == m_treeView) {
        auto *treeView = qobject_cast<QTreeView *>(watched);
        if (treeView) {
            if (event->type() == QEvent::KeyPress) {
                if (auto *keyEvent = dynamic_cast<QKeyEvent *>(event)) {
                    if (keyEvent->key() == Qt::Key_F) {
                        Json msg;
                        msg["MessageType"] = "Focus";
                        sendOptixMessage(msg);
                    }
                }
            }
        }
    }
    if (changed) {
        if (auto main = zenoApp->getMainWindow()) {
            for (DisplayWidget* view : main->viewports()) {
                if (ZOptixViewport* optxview = view->optixViewport()) {
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
            else if (zeno::str_contains(link.back(), "HDRSky2")) {
                Json msg;
                msg["MessageType"] = "HDRSky2";
                msg["Content"] = link.back();
                sendOptixMessage(msg);
            }
        }

        if (ZenoMainWindow* mainWin = zenoApp->getMainWindow()) {
            mainWin->onPrimitiveSelected({object_name});
        }
    });
}

void zenooutline::sendOptixMessage(Json &msg) {
    if (auto main = zenoApp->getMainWindow()) {
        for (DisplayWidget* view : main->viewports()) {
            if (ZOptixViewport* optxview = view->optixViewport()) {
                QString msg_str = QString::fromStdString(msg.dump());
                emit optxview->sig_sendOptixMessage(msg_str);
            }
        }
    }

}
