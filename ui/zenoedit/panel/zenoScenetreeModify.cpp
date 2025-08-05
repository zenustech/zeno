#include "zenoScenetreeModify.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/zoptixviewport.h"
#include "viewport/displaywidget.h"
#include "nodesview/zenographseditor.h"
#include "nodesys/zenosubgraphscene.h"
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>

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

scenetreeModifyModel::scenetreeModifyModel(QObject* parent) : QAbstractTableModel(parent)
{
	m_items.append({ "1", "1", "1","1","1" });
	m_items.append({ "5", "5", "5", "5", "5" });
	m_items.append({ "2", "2", "2", "2", "2"});
}

scenetreeModifyModel::~scenetreeModifyModel()
{

}

int scenetreeModifyModel::rowCount(const QModelIndex& parent) const
{
	return m_items.size();
}

int scenetreeModifyModel::columnCount(const QModelIndex& parent) const
{
	return 6;
}

QVariant scenetreeModifyModel::data(const QModelIndex& index, int role) const
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

QVariant scenetreeModifyModel::headerData(int section, Qt::Orientation orientation, int role) const
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

bool scenetreeModifyModel::setData(const QModelIndex& index, const QVariant& value, int role)
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

void scenetreeModifyModel::insertRow(const QString id, const QString r0, const QString r1, const QString r2, const QString t)
{
	beginInsertRows(QModelIndex(), m_items.size(), m_items.size());
	m_items.append({ id, r0, r1, r2, t });
	endInsertRows();
}

void scenetreeModifyModel::removeRow(int row)
{
	if (row < 0 || row >= m_items.size())
		return;
	beginRemoveRows(QModelIndex(), row, row);
	m_items.remove(row);
	endRemoveRows();
}

QModelIndex& scenetreeModifyModel::indexFromId(QString id)
{
	for (int i = 0; i < m_items.size(); ++i) {
		if (m_items[i].id == id) {
			return createIndex(i, 0);
		}
	}
	return QModelIndex();
}

std::vector<std::string> scenetreeModifyModel::getRow(int row) const
{
	if (row < 0 || row > m_items.size()) {
		return {};
	}
	auto item = m_items[row];
	return { item.id.toStdString(), item.r0.toStdString(), item.r1.toStdString(), item.r2.toStdString(), item.t.toStdString() };
}

void scenetreeModifyModel::setupModelDataFromMessage(Json const& content)
{
	beginResetModel();

	m_items.clear();
	for (auto& value : content["id"]) {
		m_items.append({ QString::fromStdString(value["id"]), 
			QString::fromStdString(value["r0"]) , 
			QString::fromStdString(value["r1"]) , 
			QString::fromStdString(value["r2"]) ,
			QString::fromStdString(value["t"]) });
	}

	endResetModel();
}

zenoScenetreeModify::zenoScenetreeModify(QWidget *parent)
	: QWidget(parent), m_tableView(new QTableView(this)), m_model(new scenetreeModifyModel(this))
{
	initUi();

	if (auto main = zenoApp->getMainWindow()) {
		for (DisplayWidget* view : main->viewports()) {
			if (auto optxview = view->optixViewport()) {
				connect(optxview, &ZOptixViewport::sig_viewportSendToOutline, this, [this](QString content) {//处理光追发过来的消息
					if (this->m_model) {
						Json msg = Json::parse(content.toStdString());
						if (msg["MessageType"] == "SceneTreeModification") {//初始化
							m_model->setupModelDataFromMessage(msg);
						} else if (msg["MessageType"] == "SceneTreeObjModifyInfo") {//平移旋转缩放某个物体后新增/修改条目
							auto idx = m_model->indexFromId("物体的id");
							if (idx.isValid()) {
								int r = idx.row();
								m_model->setData(m_model->index(r, 1), "r0");
								m_model->setData(m_model->index(r, 2), "r1");
								m_model->setData(m_model->index(r, 3), "r2");
								m_model->setData(m_model->index(r, 4), "t");
							}
							else {
								m_model->insertRow("id", "r0", "r1", "r2", "t");
							}
						}
					}
				});
				connect(m_tableView, &QTableView::clicked, this, [this, optxview](const QModelIndex& index) {//点击删除modification
					if (index.column() == 5) {
						Json msg;
						msg["MessageType"] = "resetObjectModify";
						msg["objectId"] = m_model->data(m_model->index(index.row(), 0)).toString().toStdString();

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
}

zenoScenetreeModify::~zenoScenetreeModify()
{

}

void zenoScenetreeModify::initUi()
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
		for (int i = 0; i < m_model->rowCount(); i++) {
			std::vector<std::string> rowdata = m_model->getRow(i);
			std::string key = rowdata[0];
			rowdata.erase(rowdata.begin());
			if (!rowdata.empty()) {
				msg[key] = rowdata;
			}
		}

		//生成/修改节点
		QString outnode("ee371442-CreateCube"), outsock("prim"), innode("SetNodeXform"), insock("scene"), inModifyInfoSock("node");
		generateModificationNode(outnode, outsock, innode, insock, inModifyInfoSock, msg);
	});
}

void zenoScenetreeModify::generateModificationNode(QString outNodeId, QString outSock, QString inNodeType, QString inSock, QString inModifyInfoSock, Json& msg)
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
		INPUT_SOCKETS inputs = existModifyNode.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
		INPUT_SOCKET input = inputs[inModifyInfoSock];

		PARAM_UPDATE_INFO info;
		info.name = inModifyInfoSock;
		info.newValue = QString::fromStdString(msg.dump());
		info.oldValue = input.info.defaultValue;
		pModel->updateSocketDefl(existModifyNode.data(ROLE_OBJID).toString(), info, subgIdx);
	}
	else {
		QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
		STATUS_UPDATE_INFO info;
		info.oldValue = pos;
		pos.setX(pos.x() + 500);
		info.newValue = pos;
		info.role = ROLE_OBJPOS;

		QString newNodeident = NodesMgr::createNewNode(pModel, subgIdx, inNodeType, QPointF(0, 0));
		pModel->updateNodeStatus(newNodeident, info, subgIdx, false);

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
		pModel->updateNodeStatus(nodeIdx.data(ROLE_OBJID).toString(), info, subgIdx);

		options = newNodeIdx.data(ROLE_OPTIONS).toInt();
		info.oldValue = options;
		options |= OPT_VIEW;
		info.newValue = options;
		pModel->updateNodeStatus(newNodeIdx.data(ROLE_OBJID).toString(), info, subgIdx);

		INPUT_SOCKETS inputs = newNodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
		INPUT_SOCKET input = inputs[inModifyInfoSock];
		PARAM_UPDATE_INFO sockinfo;
		sockinfo.name = inModifyInfoSock;
		sockinfo.newValue = QString::fromStdString(msg.dump());
		sockinfo.oldValue = input.info.defaultValue;
		pModel->updateSocketDefl(newNodeIdx.data(ROLE_OBJID).toString(), sockinfo, subgIdx);
	}
}
