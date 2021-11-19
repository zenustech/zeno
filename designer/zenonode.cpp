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
	, m_holder_nodename(nullptr)
	, m_holder_status(nullptr)
	, m_holder_control(nullptr)
	, m_holder_display(nullptr)
	, m_holder_header_backboard(nullptr)
	, m_holder_topleftsocket(nullptr)
	, m_holder_body_backboard(nullptr)
{
	pScene->addItem(this);
	//setFlags(ItemIsMovable | ItemSendsGeometryChanges | ItemIsSelectable | ItemClipsToShape);
}

void ZenoNode::initStyle(const NodeParam& param)
{
	m_param = param;

	Component& comp = m_param.header.name;
	m_holder_nodename = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);
	
	//m_holder_nodename->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\Header_back_board.jpg")));
	//m_holder_nodename->setCoreItem(new ResizableEclipseItem(QRectF(0,0,200,50)));
	m_holder_nodename->setCoreItem(new ResizableTextItem("Node Name"));

	comp = m_param.header.status;
	m_holder_status = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	//Component
	comp = m_param.header.backborad;
	m_holder_header_backboard = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);
	m_holder_header_backboard->setZValue(-10);

	comp = m_param.header.display;
	m_holder_display = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.header.control;
	m_holder_control = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.leftTopSocket;
	m_holder_topleftsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.leftBottomSocket;
	m_holder_bottomleftsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.rightTopSocket;
	m_holder_toprightsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.rightBottomSocket;
	m_holder_bottomrightsocket = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.backboard;
	m_holder_body_backboard = new ResizableItemImpl(comp.x, comp.y, comp.w, comp.h, this);
	m_holder_body_backboard->setZValue(-10);

	m_holder_status->showBorder(true);
	//element
	auto m_view = new ResizableItemImpl(20, 20, 64, 64, m_holder_status);
	m_view->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\view.jpg")));

	auto m_once = new ResizableItemImpl(80, 20, 64, 64, m_holder_status);
	m_once->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\once.jpg")));

	auto m_mute = new ResizableItemImpl(140, 20, 64, 64, m_holder_status);
	m_mute->setCoreItem(new ResizablePixmapItem(QPixmap("C:\\editor\\uirender\\mute.jpg")));
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