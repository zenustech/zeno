#include "ZenoSceneTreeModify.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/zoptixviewport.h"
#include "viewport/displaywidget.h"
#include "nodesview/zenographseditor.h"
#include "nodesys/zenosubgraphscene.h"
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>
#include <zeno/utils/string.h>
#include <zenomodel/include/command.h>

using Json = nlohmann::json;

ResetIconDelegate::ResetIconDelegate(QObject* parent) : QStyledItemDelegate(parent)
{

}

void ResetIconDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	QIcon icon = QApplication::style()->standardIcon(QStyle::SP_TitleBarCloseButton);
	if (!icon.isNull()) {
		QRect iconRect = option.rect;
		iconRect.adjust(6, 6, -6, -6);
		icon.paint(painter, iconRect, Qt::AlignCenter);
	}
	else {
		QStyledItemDelegate::paint(painter, option, index);
	}
}

SceneTreeModifyModel::SceneTreeModifyModel(QObject* parent) : QAbstractTableModel(parent)
{
//	m_items.append({ "1", "1", "1","1","1" });
//	m_items.append({ "5", "5", "5", "5", "5" });
//	m_items.append({ "2", "2", "2", "2", "2"});
}

SceneTreeModifyModel::~SceneTreeModifyModel()
{

}

int SceneTreeModifyModel::rowCount(const QModelIndex& parent) const
{
	return m_items.size();
}

int SceneTreeModifyModel::columnCount(const QModelIndex& parent) const
{
	return 6;
}

QVariant SceneTreeModifyModel::data(const QModelIndex& index, int role) const
{
	if (!index.isValid() || index.row() >= m_items.size() || index.row() < 0 || index.column() < 0 || index.column() >= 6)
		return QVariant();

	const ModifyItem& item = m_items[index.row()];
	if (role == Qt::DisplayRole) {
		switch (index.column()) {
		case 0: return item.id;
		case 1: return item.r0;
		case 2: return item.r1;
		case 3: return item.r2;
		case 4: return item.t;
		case 5: return QVariant();
		}
	}
	return QVariant();
}

QVariant SceneTreeModifyModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole)
		return QVariant();

	if (orientation == Qt::Horizontal) {
		switch (section) {
		case 0: return "id";
		case 1: return "r0";
		case 2: return "r1";
		case 3: return "r2";
		case 4: return "t";
		case 5: return "reset";
		}
	}
	else if (orientation == Qt::Vertical) {
		return section;
	}
	return QVariant();
}

bool SceneTreeModifyModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (!index.isValid() || index.row() < 0 || index.row() >= m_items.size() || index.column() < 0 || index.column() >= 5)
		return false;

	if (role == Qt::EditRole) {
		ModifyItem& item = m_items[index.row()];
		switch (index.column()) {
		case 0: item.id = value.toString(); break;
		case 1: item.r0 = value.toString(); break;
		case 2: item.r1 = value.toString(); break;
		case 3: item.r2 = value.toString(); break;
		case 4: item.t = value.toString(); break;
		default: return false;
		}
		emit dataChanged(index, index, { role });
		return true;
	}
	return false;
}

void SceneTreeModifyModel::insertRow(const QString id, const QString r0, const QString r1, const QString r2, const QString t)
{
	beginInsertRows(QModelIndex(), m_items.size(), m_items.size());
	m_items.append({ id, r0, r1, r2, t });
	endInsertRows();
}

void SceneTreeModifyModel::removeRow(int row)
{
	if (row < 0 || row >= m_items.size())
		return;
	beginRemoveRows(QModelIndex(), row, row);
	m_items.remove(row);
	endRemoveRows();
}

QModelIndex& SceneTreeModifyModel::indexFromId(QString id)
{
	for (int i = 0; i < m_items.size(); ++i) {
		if (m_items[i].id == id) {
			return createIndex(i, 0);
		}
	}
	return QModelIndex();
}

std::vector<std::string> SceneTreeModifyModel::getRow(int row) const
{
	if (row < 0 || row > m_items.size()) {
		return {};
	}
	auto item = m_items[row];
	return { item.id.toStdString(), item.r0.toStdString(), item.r1.toStdString(), item.r2.toStdString(), item.t.toStdString() };
}

void SceneTreeModifyModel::setupModelDataFromMessage(Json const& content)
{
	beginResetModel();

	m_items.clear();

    auto matrixs = content["Matrixs"];

    for (auto& [key, _mat] : matrixs.items()) {
        auto mat = _mat[0];
        m_items.append({
            QString::fromStdString(key)
            , QString::fromStdString(zeno::format("{} {} {}", float(mat[0]), float(mat[1]), float(mat[2])))
            , QString::fromStdString(zeno::format("{} {} {}", float(mat[3]), float(mat[4]), float(mat[5])))
            , QString::fromStdString(zeno::format("{} {} {}", float(mat[6]), float(mat[7]), float(mat[8])))
            , QString::fromStdString(zeno::format("{} {} {}", float(mat[9]), float(mat[10]), float(mat[11])))
        });
    }

	endResetModel();
}

ZenoSceneTreeModify::ZenoSceneTreeModify(QWidget *parent)
	: QWidget(parent), m_tableView(new QTableView(this)), m_model(new SceneTreeModifyModel(this))
{
	initUi();

	if (auto main = zenoApp->getMainWindow()) {
		for (DisplayWidget* view : main->viewports()) {
			if (auto optxview = view->optixViewport()) {
				connect(optxview, &ZOptixViewport::sig_viewportSendToXformPanel, this, [this](QString content) {//处理光追发过来的消息
					if (this->m_model) {
						Json msg = Json::parse(content.toStdString());
						if (msg["MessageType"] == "XformPanelInitFeedback") {//初始化
							m_model->setupModelDataFromMessage(msg);
							multistring_uuid.clear();
						} else if (msg["MessageType"] == "SetNodeXform") {
                            auto node_name = QString::fromStdString(std::string(msg["NodeName"]));
                            auto r0 = QString::fromStdString(zeno::format("{}, {}, {}", float(msg["r0"][0]), float(msg["r0"][1]), float(msg["r0"][2])));
                            auto r1 = QString::fromStdString(zeno::format("{}, {}, {}", float(msg["r1"][0]), float(msg["r1"][1]), float(msg["r1"][2])));
                            auto r2 = QString::fromStdString(zeno::format("{}, {}, {}", float(msg["r2"][0]), float(msg["r2"][1]), float(msg["r2"][2])));
                            auto  t = QString::fromStdString(zeno::format("{}, {}, {}", float(msg[ "t"][0]), float(msg[ "t"][1]), float(msg[ "t"][2])));
							auto idx = m_model->indexFromId(node_name);
							if (idx.isValid()) {
								int r = idx.row();
								m_model->setData(m_model->index(r, 1), r0);
								m_model->setData(m_model->index(r, 2), r1);
								m_model->setData(m_model->index(r, 3), r2);
								m_model->setData(m_model->index(r, 4), t);
							}
							else {
								m_model->insertRow(node_name, r0, r1, r2, t);
							}
						} else if (msg["MessageType"] == "SetSceneXform") {
							auto content_std = content.toStdString();
							Json json = Json::parse(content_std);
							auto node_key = std::string(json["NodeKey"]);
							auto cur_node_uuid = zeno::split_str(node_key, ':')[0];

							std::string outnode = cur_node_uuid;
							//generateModificationNode(outnode, "scene", "MakeMultilineString", "scene", "value", json["Matrixs"]);
							generateModificationNode(QString::fromStdString(outnode), "scene", "SetSceneXform", "scene", "xformsList", json["Matrixs"]);
						}
					}
				});
				connect(m_tableView, &QTableView::clicked, this, [this, optxview](const QModelIndex& index) {//点击删除modification
					if (index.column() == 5) {
						Json msg;
						msg["MessageType"] = "ResetNodeModify";
						msg["NodeName"] = m_model->data(m_model->index(index.row(), 0)).toString().toStdString();

						m_model->removeRow(index.row());

						emit optxview->sig_sendOptixMessage(QString::fromStdString(msg.dump()));
					}
				});

				Json msg;
				msg["MessageType"] = "InitStModifyPanel";
				emit optxview->sig_sendOptixMessage(QString::fromStdString(msg.dump()));//发送初始化信号
			}
		}
	}
    Json msg;
    msg["MessageType"] = "XformPanelInit";
    sendOptixMessage(msg);
}

ZenoSceneTreeModify::~ZenoSceneTreeModify()
{

}

void ZenoSceneTreeModify::initUi()
{
	QSortFilterProxyModel* proxyModel = new QSortFilterProxyModel(this);
	proxyModel->setSourceModel(m_model);
	proxyModel->setSortCaseSensitivity(Qt::CaseInsensitive); // 可选

	m_tableView->setModel(proxyModel);
	m_tableView->setSortingEnabled(true);
	m_tableView->sortByColumn(0, Qt::AscendingOrder);

	m_tableView->setEditTriggers(QAbstractItemView::NoEditTriggers);
	m_tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	m_tableView->setItemDelegateForColumn(5, new ResetIconDelegate(this));

	QVBoxLayout* mainlayout = new QVBoxLayout(this);

	QHBoxLayout* ptitalLayout = new QHBoxLayout(this);
	QPushButton* synctonode = new QPushButton("Sync to node");
	ptitalLayout->addStretch();
	ptitalLayout->addWidget(synctonode);

	mainlayout->addLayout(ptitalLayout);
	mainlayout->addWidget(m_tableView);
	mainlayout->setContentsMargins(0, 0, 0, 0);
	setLayout(mainlayout);

	connect(synctonode, &QPushButton::clicked, this, [this]() {
		Json msg;
        msg["MessageType"] = "NeedSetSceneXform";
        sendOptixMessage(msg);
	});
}

void ZenoSceneTreeModify::generateModificationNode(QString outNodeId, QString outSock, QString inNodeType, QString inSock, QString inModifyInfoSock, Json& msg)
{
	auto main = zenoApp->getMainWindow();
	ZASSERT_EXIT(main);
	auto editor = main->getAnyEditor();
	ZASSERT_EXIT(editor);
	auto view = editor->getCurrentSubGraphView();
	ZASSERT_EXIT(view);
	auto scene = view->scene();
	ZASSERT_EXIT(scene);

	IGraphsModel* pModel = GraphsManagment::instance().currentModel();
	ZASSERT_EXIT(pModel);
	QModelIndex subgIdx = scene->subGraphIndex();
	ZASSERT_EXIT(subgIdx.isValid());

	if (multistring_uuid.size()) {
		QModelIndex multilinestrIdx = pModel->index(QString::fromStdString(multistring_uuid), subgIdx);
		PARAMS_INFO params = multilinestrIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
		PARAM_INFO param = params["value"];
		PARAM_UPDATE_INFO paraminfo;
		paraminfo.name = "value";
		paraminfo.newValue = QString::fromStdString(msg.dump());
		paraminfo.oldValue = param.value;
		pModel->updateParamInfo(multilinestrIdx.data(ROLE_OBJID).toString(), paraminfo, subgIdx);

		return;
	}

	if (outNodeId.contains("SetSceneXform"))
	{
		QModelIndex xformIdx = pModel->index(outNodeId, subgIdx);
		multistring_uuid = addMultilineStr(xformIdx, msg, inModifyInfoSock);
		return;
	}

	QModelIndex nodeIdx = pModel->index(outNodeId, subgIdx);
	ZASSERT_EXIT(nodeIdx.isValid());

	QModelIndex existModifyNode;
	OUTPUT_SOCKETS outputs = nodeIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
	OUTPUT_SOCKET output = outputs[outSock];
	for (auto link : output.info.links)
	{
		QString inNode = UiHelper::getSockNode(link.inSockPath);
		QModelIndex inIdx = pModel->index(inNode, subgIdx);
		if (inIdx.data(ROLE_OBJNAME).toString() == inNodeType) {
			existModifyNode = inIdx;
			break;
		}
	}
	if (existModifyNode.isValid()) {
		multistring_uuid = addMultilineStr(existModifyNode, msg, inModifyInfoSock);
	}
	else {
		QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
		STATUS_UPDATE_INFO info;
		info.oldValue = pos;
		pos.setX(pos.x() + 600);
		info.newValue = pos;
		info.role = ROLE_OBJPOS;

		QString newNodeident = NodesMgr::createNewNode(pModel, subgIdx, inNodeType, pos);
		//pModel->updateNodeStatus(newNodeident, info, subgIdx, false);

		QModelIndex newNodeIdx = pModel->index(newNodeident, subgIdx);
		NodeParamModel* inNodeParams = QVariantPtr<NodeParamModel>::asPtr(newNodeIdx.data(ROLE_NODE_PARAMS));
		QModelIndex inSockIdx = inNodeParams->getParam(PARAM_INPUT, inSock);
		NodeParamModel* outNodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
		QModelIndex outSockIdx = outNodeParams->getParam(PARAM_OUTPUT, outSock);
		pModel->addLink(subgIdx, outSockIdx, inSockIdx);

		STATUS_UPDATE_INFO viewinfo;
		int options = nodeIdx.data(ROLE_OPTIONS).toInt();
		info.role = ROLE_OPTIONS;
		info.oldValue = options;
		options = options & (~OPT_VIEW);
		info.newValue = options;
		pModel->updateNodeStatus(nodeIdx.data(ROLE_OBJID).toString(), info, subgIdx);//关掉旧节点的view

		options = newNodeIdx.data(ROLE_OPTIONS).toInt();
		info.oldValue = options;
		options |= OPT_VIEW;
		info.newValue = options;
		pModel->updateNodeStatus(newNodeIdx.data(ROLE_OBJID).toString(), info, subgIdx);//新节点开view

		//multilinestr节点
		multistring_uuid = addMultilineStr(newNodeIdx, msg, inModifyInfoSock);
	}
}

std::string ZenoSceneTreeModify::addMultilineStr(QModelIndex newNodeIdx, Json& msg, QString inModifyInfoSock)
{
	auto main = zenoApp->getMainWindow();
	ZASSERT_EXIT(main, "");
	auto editor = main->getAnyEditor();
	ZASSERT_EXIT(editor, "");
	auto view = editor->getCurrentSubGraphView();
	ZASSERT_EXIT(view, "");
	auto scene = view->scene();
	ZASSERT_EXIT(scene, "");

	IGraphsModel* pModel = GraphsManagment::instance().currentModel();
	ZASSERT_EXIT(pModel, "");
	QModelIndex subgIdx = scene->subGraphIndex();
	ZASSERT_EXIT(subgIdx.isValid(), "");

	NodeParamModel* newNodeIdxParams = QVariantPtr<NodeParamModel>::asPtr(newNodeIdx.data(ROLE_NODE_PARAMS));
	QModelIndex newNodeIdxSockIdx = newNodeIdxParams->getParam(PARAM_INPUT, inModifyInfoSock);

	QAbstractItemModel* pKeyObjModel = QVariantPtr<QAbstractItemModel>::asPtr(newNodeIdxSockIdx.data(ROLE_VPARAM_LINK_MODEL));
	int n = pKeyObjModel->rowCount();

	QPointF pos = newNodeIdx.data(ROLE_OBJPOS).toPointF();
	QString multilinestrNodeident = NodesMgr::createNewNode(pModel, subgIdx, "MakeMultilineString", { pos.x() - 300, pos.y() + (n + 1) * 300 });

	QModelIndex multilinestrIdx = pModel->index(multilinestrNodeident, subgIdx);
	NodeParamModel* multilinestrIdxParams = QVariantPtr<NodeParamModel>::asPtr(multilinestrIdx.data(ROLE_NODE_PARAMS));
	QModelIndex multilinestrSockIdx = multilinestrIdxParams->getParam(PARAM_OUTPUT, "value");

	pModel->addExecuteCommand(new DictKeyAddRemCommand(true, pModel, newNodeIdxSockIdx.data(ROLE_OBJPATH).toString(), n));
	pModel->addLink(subgIdx, multilinestrSockIdx, pKeyObjModel->index(n, 0));

	PARAMS_INFO params = multilinestrIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
	PARAM_INFO param = params["value"];
	PARAM_UPDATE_INFO paraminfo;
	paraminfo.name = "value";
	paraminfo.newValue = QString::fromStdString(msg.dump());
	paraminfo.oldValue = param.value;
	pModel->updateParamInfo(multilinestrIdx.data(ROLE_OBJID).toString(), paraminfo, subgIdx);

	return multilinestrNodeident.toStdString();
}

void ZenoSceneTreeModify::sendOptixMessage(Json &msg) {
    if (auto main = zenoApp->getMainWindow()) {
        for (DisplayWidget* view : main->viewports()) {
            if (ZOptixViewport* optxview = view->optixViewport()) {
                QString msg_str = QString::fromStdString(msg.dump());
                emit optxview->sig_sendOptixMessage(msg_str);
            }
        }
    }
}
