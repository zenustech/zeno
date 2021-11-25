#include "framework.h"
#include "nodetemplate.h"
#include "nodescene.h"
#include "resizableitemimpl.h"
#include "resizecoreitem.h"
#include <rapidjson/document.h>

using namespace rapidjson;


NodeTemplate::NodeTemplate(NodeScene* pScene, QGraphicsItem* parent)
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

void NodeTemplate::initStyle(const NodeParam& param)
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

QStandardItem* NodeTemplate::createItemWithGVItem(ResizableItemImpl* gvItem, NODE_ID id, const QString& name, QStandardItemModel* pModel, QItemSelectionModel* selection)
{
    QStandardItem* pItem = new QStandardItem(QIcon(), name);
    pItem->setData(id);
    pItem->setData(gvItem->coreItemSceneRect(), NODEPOS_ROLE);
	pItem->setData(true, NODELOCK_ROLE);
	pItem->setData(false, NODELOCK_VISIBLE);
    connect(gvItem, &ResizableItemImpl::itemGeoChanged, this, [=](QRectF rcNew) {
        pItem->setData(rcNew, NODEPOS_ROLE);
        });
    connect(gvItem, &ResizableItemImpl::itemDeselected, this, [=]() {
        selection->select(pItem->index(), QItemSelectionModel::Deselect);
        });
	connect(pModel, &QStandardItemModel::itemChanged, this, [=](QStandardItem* pItemChanged) {
			if (pItemChanged == pItem)
			{
				QRectF rc = pItemChanged->data(NODEPOS_ROLE).toRectF();
				gvItem->setCoreItemSceneRect(rc);
			}
		});
	return pItem;
}

QVariant NodeTemplate::itemChange(GraphicsItemChange change, const QVariant& value)
{
	return QGraphicsObject::itemChange(change, value);
}

void NodeTemplate::initModel(QStandardItemModel* pModel, QItemSelectionModel* selection)
{
	if (!pModel)
		return;

	pModel->clear();

	QStandardItem* headerItem = new QStandardItem(QIcon(), "Header");
	headerItem->setData(NODE_ID::HEADER);
	if (m_component_nodename)
	{
		QStandardItem* nodenameItem = createItemWithGVItem(m_component_nodename, NODE_ID::COMP_NODENAME, "Node-name", pModel, selection);
		headerItem->appendRow(nodenameItem);
	}
	if (m_component_status)
	{
		QStandardItem* statusItem = createItemWithGVItem(m_component_status, NODE_ID::COMP_STATUS, "Status", pModel, selection);
		headerItem->appendRow(statusItem);
	}
	if (m_component_control)
	{
		QStandardItem* controlItem = createItemWithGVItem(m_component_control, NODE_ID::COMP_CONTROL, "Control", pModel, selection);
		headerItem->appendRow(controlItem);
	}
	if (m_component_header_backboard)
	{
		QStandardItem* backboardItem = createItemWithGVItem(m_component_header_backboard, NODE_ID::COMP_HEADER_BACKBOARD, "Back-board", pModel, selection);
		headerItem->appendRow(backboardItem);
	}
	if (m_component_display)
	{
		QStandardItem* displayItem = createItemWithGVItem(m_component_display, NODE_ID::COMP_DISPLAY, "Display", pModel, selection);
		headerItem->appendRow(displayItem);
	}

	QStandardItem* bodyItem = new QStandardItem(QIcon(), "Body");
	bodyItem->setData(NODE_ID::BODY);
	if (m_component_ltsocket)
	{
		QStandardItem* ltsocketItem = createItemWithGVItem(m_component_ltsocket, NODE_ID::COMP_LTSOCKET, "LTSocket", pModel, selection);
		bodyItem->appendRow(ltsocketItem);
	}
	if (m_component_lbsocket)
	{
		QStandardItem* lbsocketItem = createItemWithGVItem(m_component_lbsocket, NODE_ID::COMP_LBSOCKET, "LBSocket", pModel, selection);
		bodyItem->appendRow(lbsocketItem);
	}
	if (m_component_rtsocket)
	{
		QStandardItem* rtsocketItem = createItemWithGVItem(m_component_rtsocket, NODE_ID::COMP_RTSOCKET, "RTSocket", pModel, selection);
		bodyItem->appendRow(rtsocketItem);
	}
	if (m_component_rbsocket)
	{
        QStandardItem* rbsocketItem = createItemWithGVItem(m_component_rbsocket, NODE_ID::COMP_RBSOCKET, "RBSocket", pModel, selection);
        bodyItem->appendRow(rbsocketItem);
	}
	if (m_component_body_backboard)
	{
		QStandardItem* backboardItem = createItemWithGVItem(m_component_body_backboard, NODE_ID::COMP_BODYBACKBOARD, "Back-board", pModel, selection);
		bodyItem->appendRow(backboardItem);
	}

	pModel->appendRow(headerItem);
	pModel->appendRow(bodyItem);
}

void NodeTemplate::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected)
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
			m_component_nodename->setZValue(100);
			break;

		case COMP_STATUS:
			m_component_status->setSelected(true);
			m_component_status->setZValue(100);
			break;

		case COMP_CONTROL:
			m_component_control->setSelected(true);
			m_component_control->setZValue(100);
			break;

		case COMP_DISPLAY:
			m_component_display->setSelected(true);
			m_component_display->setZValue(100);
			break;

		case COMP_HEADER_BACKBOARD:
			m_component_header_backboard->setSelected(true);
			m_component_header_backboard->setZValue(100);
			break;

		case COMP_BODYBACKBOARD:
			m_component_body_backboard->setSelected(true);
			m_component_body_backboard->setZValue(100);
			break;

		case COMP_LTSOCKET:
			m_component_ltsocket->setSelected(true);
			m_component_ltsocket->setZValue(100);
			break;

		case COMP_LBSOCKET:
			m_component_lbsocket->setSelected(true);
			m_component_lbsocket->setZValue(100);
			break;

		case COMP_RTSOCKET:
			m_component_rtsocket->setSelected(true);
			m_component_rtsocket->setZValue(100);
			break;

		case COMP_RBSOCKET:
			m_component_rbsocket->setSelected(true);
			m_component_rbsocket->setZValue(100);
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
			m_component_nodename->setZValue(-100);
			break;

		case COMP_STATUS:
			m_component_status->setSelected(false);
			m_component_status->setZValue(-100);
			break;

		case COMP_CONTROL:
			m_component_control->setSelected(false);
			m_component_control->setZValue(-100);
			break;

		case COMP_DISPLAY:
			m_component_display->setSelected(false);
			m_component_display->setZValue(-100);
			break;

		case COMP_HEADER_BACKBOARD:
			m_component_header_backboard->setSelected(false);
			m_component_header_backboard->setZValue(-100);
			break;

		case COMP_BODYBACKBOARD:
			m_component_body_backboard->setSelected(false);
			m_component_body_backboard->setZValue(-100);
			break;

		case COMP_LTSOCKET:
			m_component_ltsocket->setSelected(false);
			m_component_ltsocket->setZValue(-100);
			break;

		case COMP_LBSOCKET:
			m_component_lbsocket->setSelected(false);
			m_component_lbsocket->setZValue(-100);
			break;

		case COMP_RTSOCKET:
			m_component_rtsocket->setSelected(false);
			m_component_rtsocket->setZValue(-100);
			break;

		case COMP_RBSOCKET:
			m_component_rbsocket->setSelected(false);
			m_component_rbsocket->setZValue(-100);
			break;
		}
	}
}

QRectF NodeTemplate::boundingRect() const
{
	QRectF wtf = this->childrenBoundingRect();
	return wtf;
}

QPainterPath NodeTemplate::shape() const
{
	return QGraphicsObject::shape();
}

void NodeTemplate::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}