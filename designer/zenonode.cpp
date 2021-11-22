#include "framework.h"
#include "zenonode.h"
#include "nodescene.h"
#include "resizableitemimpl.h"
#include "resizecoreitem.h"
#include <rapidjson/document.h>

using namespace rapidjson;


ZenoNode::ZenoNode(NodeScene* pScene, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_once(nullptr)
	, m_prep(nullptr)
	, m_mute(nullptr)
	, m_view(nullptr)
	, m_genshin(nullptr)
	, m_background(nullptr)
	, m_nodename(nullptr)
	, m_component_nodename(nullptr)
	, m_component_status(nullptr)
	, m_component_control(nullptr)
	, m_component_display(nullptr)
	, m_component_header_backboard(nullptr)
	, m_component_ltsocket(nullptr)
	, m_component_body_backboard(nullptr)
{
	pScene->addItem(this);
	//setFlags(ItemIsMovable | ItemSendsGeometryChanges | ItemIsSelectable | ItemClipsToShape);
}

void ZenoNode::initStyle(const NodeParam& param)
{
	m_param = param;

	

	Component& comp = m_param.header.name;
	m_component_nodename = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);
	
	//m_holder_nodename->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\Header_back_board.jpg")));
	//m_holder_nodename->setCoreItem(new ResizableEclipseItem(QRectF(0,0,200,50)));
	//m_holder_nodename->setCoreItem(new ResizableTextItem("Node Name"));

	comp = m_param.header.status;
	m_component_status = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	//Component
	comp = m_param.header.backborad;
	m_component_header_backboard = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);
	m_component_header_backboard->setZValue(-10);

	comp = m_param.header.display;
	m_component_display = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.header.control;
	m_component_control = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.leftTopSocket;
	m_component_ltsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.leftBottomSocket;
	m_component_lbsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.rightTopSocket;
	m_component_rtsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.rightBottomSocket;
	m_component_rbsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.backboard;
	m_component_body_backboard = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);
	m_component_body_backboard->setZValue(-10);

	m_component_status->showBorder(true);

	//element
	//auto m_view = new ResizableItemImpl(20, 20, 64, 64, m_holder_status);
	//m_view->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\view.jpg")));

	//auto m_once = new ResizableItemImpl(80, 20, 64, 64, m_holder_status);
	//m_once->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\once.jpg")));

	//auto m_mute = new ResizableItemImpl(140, 20, 64, 64, m_holder_status);
	//m_mute->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\mute.jpg")));
}

void ZenoNode::initModel(QStandardItemModel* pModel)
{
	if (!pModel)
		return;

	pModel->clear();

	QStandardItem* headerItem = new QStandardItem(QIcon(), "Header");
	headerItem->setData(NODE_ID::HEADER);
	if (m_component_nodename)
	{
		QStandardItem* nodenameItem = new QStandardItem(QIcon(), "Node-name");
		nodenameItem->setData(NODE_ID::COMP_NODENAME);
		headerItem->appendRow(nodenameItem);
	}
	if (m_component_status)
	{
		QStandardItem* statusItem = new QStandardItem(QIcon(), "Status");
		statusItem->setData(NODE_ID::COMP_STATUS);
		headerItem->appendRow(statusItem);
	}
	if (m_component_control)
	{
		QStandardItem* controlItem = new QStandardItem(QIcon(), "Control");
		controlItem->setData(NODE_ID::COMP_CONTROL);
		headerItem->appendRow(controlItem);
	}
	if (m_component_header_backboard)
	{
		QStandardItem* backboardItem = new QStandardItem(QIcon(), "Back-board");
		backboardItem->setData(NODE_ID::COMP_HEADER_BACKBOARD);
		headerItem->appendRow(backboardItem);
	}
	if (m_component_display)
	{
		QStandardItem* displayItem = new QStandardItem(QIcon(), "Display");
		displayItem->setData(NODE_ID::COMP_DISPLAY);
		headerItem->appendRow(displayItem);
	}

	QStandardItem* bodyItem = new QStandardItem(QIcon(), "Body");
	bodyItem->setData(NODE_ID::BODY);
	if (m_component_ltsocket)
	{
		QStandardItem* ltsocketItem = new QStandardItem(QIcon(), "LTSocket");
		ltsocketItem->setData(NODE_ID::COMP_LTSOCKET);
		bodyItem->appendRow(ltsocketItem);
	}
	if (m_component_lbsocket)
	{
		QStandardItem* lbsocketItem = new QStandardItem(QIcon(), "LBSocket");
		lbsocketItem->setData(NODE_ID::COMP_LBSOCKET);
		bodyItem->appendRow(lbsocketItem);
	}
	if (m_component_rtsocket)
	{
		QStandardItem* rtsocketItem = new QStandardItem(QIcon(), "RTSocket");
		rtsocketItem->setData(NODE_ID::COMP_RTSOCKET);
		bodyItem->appendRow(rtsocketItem);
	}
	if (m_component_rbsocket)
	{
		QStandardItem* rbsocketItem = new QStandardItem(QIcon(), "RBSocket");
		rbsocketItem->setData(NODE_ID::COMP_RBSOCKET);
		bodyItem->appendRow(rbsocketItem);
	}

	if (m_component_body_backboard)
	{
		QStandardItem* backboardItem = new QStandardItem(QIcon(), "Back-board");
		backboardItem->setData(NODE_ID::COMP_BODYBACKBOARD);
		bodyItem->appendRow(backboardItem);
	}

	pModel->appendRow(headerItem);
	pModel->appendRow(bodyItem);
}

void ZenoNode::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected)
{
	QModelIndexList lst = selected.indexes();
	if (!lst.isEmpty())
	{
		QModelIndex idx = lst.at(0);
		NODE_ID id = (NODE_ID)idx.data(Qt::UserRole + 1).toInt();
		switch (id)
		{
		case COMP_NODENAME:
			m_component_nodename->setSelected(true);
			break;

		case COMP_STATUS:
			m_component_status->setSelected(true);
			break;

		case COMP_CONTROL:
			m_component_control->setSelected(true);
			break;

		case COMP_DISPLAY:
			m_component_display->setSelected(true);
			break;

		case COMP_HEADER_BACKBOARD:
			m_component_header_backboard->setSelected(true);
			break;

		case COMP_BODYBACKBOARD:
			m_component_body_backboard->setSelected(true);
			break;

		case COMP_LTSOCKET:
			m_component_ltsocket->setSelected(true);
			break;

		case COMP_LBSOCKET:
			m_component_lbsocket->setSelected(true);
			break;

		case COMP_RTSOCKET:
			m_component_rtsocket->setSelected(true);
			break;

		case COMP_RBSOCKET:
			m_component_rbsocket->setSelected(true);
			break;
		}
	}

	lst = deselected.indexes();
	if (!lst.isEmpty())
	{
		QModelIndex idx = lst.at(0);
		NODE_ID id = (NODE_ID)idx.data(Qt::UserRole + 1).toInt();
		switch (id)
		{
		case COMP_NODENAME:
			m_component_nodename->setSelected(false);
			break;

		case COMP_STATUS:
			m_component_status->setSelected(false);
			break;

		case COMP_CONTROL:
			m_component_control->setSelected(false);
			break;

		case COMP_DISPLAY:
			m_component_display->setSelected(false);
			break;

		case COMP_HEADER_BACKBOARD:
			m_component_header_backboard->setSelected(false);
			break;

		case COMP_BODYBACKBOARD:
			m_component_body_backboard->setSelected(false);
			break;

		case COMP_LTSOCKET:
			m_component_ltsocket->setSelected(false);
			break;

		case COMP_LBSOCKET:
			m_component_lbsocket->setSelected(false);
			break;

		case COMP_RTSOCKET:
			m_component_rtsocket->setSelected(false);
			break;

		case COMP_RBSOCKET:
			m_component_rbsocket->setSelected(false);
			break;
		}
	}
}

QRectF ZenoNode::boundingRect() const
{
	QRectF wtf = this->childrenBoundingRect();
	return wtf;
}

QPainterPath ZenoNode::shape() const
{
	return QGraphicsObject::shape();
}

void ZenoNode::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	//painter->fillRect(boundingRect(), QColor(0,0,0));
}