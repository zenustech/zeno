#include "zenooutline.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/displaywidget.h"
#include <zenovis/ObjectsManager.h>
#include <tinygltf/json.hpp>
#include "viewport/zenovis.h"
#include "viewport/displaywidget.h"
#include "viewport/zoptixviewport.h"
#include <zenomodel/include/api.h>
#include <zeno/utils/string.h>

using Json = nlohmann::json;

static void add_set_node_xform(
	ZENO_HANDLE hGraph
	, std::optional<std::string>& cur_output_uuid
	, std::string const& outline_node_name
	, Json const& mat
	, std::unordered_map<std::string, std::string>& outline_node_to_uuid
) {
	zeno::log_info("outline_node_name: {}", outline_node_name);
	if (mat["Mode"] == "Set") {
		if (outline_node_to_uuid.count(outline_node_name) == 0) {
			auto node = Zeno_GetNode(hGraph, cur_output_uuid.value());
			if (!node.has_value()) {
				return;
			}

			auto new_node_handle = Zeno_AddNode(hGraph, "SetNodeXform");
			//            Zeno_SetView(hGraph, new_node_handle, true);
			//            auto& node_sync = zeno::NodeSyncMgr::GetInstance();
			//            auto node_loc = node_sync.searchNode(cur_output_uuid.value());
			//            if (node_loc.has_value()) {
			//                node_sync.updateNodeVisibility(node_loc.value());
			//                zeno::log_info("node_loc.has_value");
			//            }
			auto pos = Zeno_GetPos(hGraph, node.value());
			if (pos.has_value()) {
				zeno::vec2f npos = pos.value();
				npos += zeno::vec2f(500, 0);
				Zeno_SetPos(hGraph, new_node_handle, { npos[0], npos[1] });
			}
			std::string node_uuid;
			auto err = Zeno_GetNodeUuid(hGraph, new_node_handle, node_uuid);
			if (err != 0) {
				return;
			}
			outline_node_to_uuid[outline_node_name] = node_uuid;
			cur_output_uuid = node_uuid;
			err = Zeno_SetInputDefl(hGraph, new_node_handle, "node", outline_node_name);
			if (err != 0) {
				return;
			}
			Zeno_AddLink(hGraph, node.value(), "scene", new_node_handle, "scene");
			//            Zeno_SetView(hGraph, new_node_handle, true);
			Zeno_SetView(hGraph, node.value(), false);
		}
		auto output_uuid = outline_node_to_uuid[outline_node_name];
		auto node = Zeno_GetNode(hGraph, output_uuid);
		if (!node.has_value()) {
			return;
		}
		auto const& r0 = mat["r0"];
		auto const& r1 = mat["r1"];
		auto const& r2 = mat["r2"];
		auto const& t = mat["t"];
		Zeno_SetInputDefl(hGraph, node.value(), "r0", zeno::vec3f(r0[0], r0[1], r0[2]));
		Zeno_SetInputDefl(hGraph, node.value(), "r1", zeno::vec3f(r1[0], r1[1], r1[2]));
		Zeno_SetInputDefl(hGraph, node.value(), "r2", zeno::vec3f(r2[0], r2[1], r2[2]));
		Zeno_SetInputDefl(hGraph, node.value(), "t", zeno::vec3f(t[0], t[1], t[2]));
	}
}

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
					} else if (msg["MessageType"] == "SetNodeXform") {
						ZENO_HANDLE hGraph = Zeno_GetGraph("main");
						auto outline_node_name = std::string(msg["NodeName"]);
						if (!this->cur_node_uuid.has_value()) {
							auto node_key = std::string(msg["NodeKey"]);
							cur_node_uuid = zeno::split_str(node_key, ':')[0];
						}
						if (this->cur_node_uuid.has_value()) {
							add_set_node_xform(hGraph, this->cur_node_uuid, outline_node_name, msg, this->outline_node_to_uuid);
						}
					}
                });
            }
        }
    }
	IGraphsModel* pModel = GraphsManagment::instance().currentModel();
	if (pModel) {
		connect(pModel, &IGraphsModel::_rowsAboutToBeRemoved, this, [this, pModel](const QModelIndex& subgIdx, const QModelIndex& parent, int first, int last) {
			QModelIndex idx = pModel->index(first, subgIdx);
			QString id = idx.data(ROLE_OBJID).toString();
			if (auto main = zenoApp->getMainWindow()) {
				for (DisplayWidget* view : main->viewports()) {
                    if (ZOptixViewport* optxview = view->optixViewport()) {
                        emit optxview->sig_nodeRemoved(id);
                    }
				}
			}
		});
	}

    Json msg;
    msg["MessageType"] = "Init";
    sendOptixMessage(msg);
}

zenooutline::~zenooutline()
{
}

bool zenooutline::eventFilter(QObject *watched, QEvent *event) {
    Json msg;
    msg["MessageType"] = "Xform";
    if (watched == m_treeView) {
        auto *treeView = qobject_cast<QTreeView *>(watched);
        if (treeView) {
            if (event->type() == QEvent::KeyPress) {
                if (auto *keyEvent = dynamic_cast<QKeyEvent *>(event)) {
                    if (keyEvent->key() == Qt::Key_R) {
                        msg["Mode"] = "Reset";
                        msg["Value"] = 0.0;
                    }
                    else if(keyEvent->key() == Qt::Key_Up) {
                        msg["Mode"] = "Scale";
                        msg["Axis"] = "XYZ";
                        msg["Value"] = 1.0;
                    }
                    else if(keyEvent->key() == Qt::Key_Down) {
                        msg["Mode"] = "Scale";
                        msg["Axis"] = "XYZ";
                        msg["Value"] = -1.0;
                    }
                    else {
                        auto modifiers = keyEvent->modifiers();
                        bool shiftPressed = modifiers & Qt::ShiftModifier;
                        bool altPressed = modifiers & Qt::AltModifier;
                        if (shiftPressed == false && altPressed == false) {
                            msg["Mode"] = "Translate";
                            if(keyEvent->key() == Qt::Key_E) {
                                msg["Axis"] = "Y";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_Q) {
                                msg["Axis"] = "Y";
                                msg["Value"] = -1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_D) {
                                msg["Axis"] = "X";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_A) {
                                msg["Axis"] = "X";
                                msg["Value"] = -1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_S) {
                                msg["Axis"] = "Z";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_W) {
                                msg["Axis"] = "Z";
                                msg["Value"] = -1.0;
                            }
                        }
                        else if (shiftPressed == true && altPressed == false) {
                            msg["Mode"] = "Rotate";
                            if(keyEvent->key() == Qt::Key_E) {
                                msg["Axis"] = "Y";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_Q) {
                                msg["Axis"] = "Y";
                                msg["Value"] = -1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_D) {
                                msg["Axis"] = "X";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_A) {
                                msg["Axis"] = "X";
                                msg["Value"] = -1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_S) {
                                msg["Axis"] = "Z";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_W) {
                                msg["Axis"] = "Z";
                                msg["Value"] = -1.0;
                            }
                        }
                        else if (shiftPressed == false && altPressed == true) {
                            msg["Mode"] = "Scale";
                            if(keyEvent->key() == Qt::Key_E) {
                                msg["Axis"] = "Y";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_Q) {
                                msg["Axis"] = "Y";
                                msg["Value"] = -1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_D) {
                                msg["Axis"] = "X";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_A) {
                                msg["Axis"] = "X";
                                msg["Value"] = -1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_S) {
                                msg["Axis"] = "Z";
                                msg["Value"] = 1.0;
                            }
                            else if(keyEvent->key() == Qt::Key_W) {
                                msg["Axis"] = "Z";
                                msg["Value"] = -1.0;
                            }
                        }
                    }
                }
            }
        }
    }
    if (msg.contains("Value")) {
        sendOptixMessage(msg);
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
