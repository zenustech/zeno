#include "framework.h"
#include "zenonode.h"
#include "nodescene.h"
#include "resizerectitem.h"
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
	m_holder_nodename = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);
	QPointF pos = m_holder_nodename->scenePos();

	comp = m_param.header.status;
	m_holder_status = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.header.backborad;
	m_holder_header_backboard = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);
	m_holder_header_backboard->setZValue(-10);

	comp = m_param.header.display;
	m_holder_display = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.header.control;
	m_holder_control = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.backboard;
	m_holder_body_backboard = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);
	m_holder_body_backboard->setZValue(-10);

	comp = m_param.body.leftTop;
	m_holder_topleftsocket = new ResizableRectItem(comp.x, comp.y, comp.w, comp.h, this);

	/*
	m_nodename = new QGraphicsTextItem("Node-name", this);
	QTextDocument* doc = m_nodename->document();
	QTextFrame* rootFrame = doc->rootFrame();
	m_nodename->setDefaultTextColor(QColor(204,204,204));

	m_nodename->setFont(m_param.header.name);
	m_background = new QGraphicsPixmapItem(QPixmap(m_param.background.normal).scaled(m_param.background.w, m_param.background.h), this);
	m_once = new QGraphicsPixmapItem(QPixmap(m_param.once.normal).scaled(m_param.once.w, m_param.once.h), this);
	m_mute = new QGraphicsPixmapItem(QPixmap(m_param.mute.normal).scaled(m_param.mute.w, m_param.mute.h), this);
	m_view = new QGraphicsPixmapItem(QPixmap(m_param.view.normal).scaled(m_param.view.w, m_param.view.h), this);
	m_genshin = new QGraphicsPixmapItem(QPixmap(m_param.genshin.normal).scaled(m_param.genshin.w, m_param.genshin.h), this);

	m_nodename->setPos(m_param.nodename.x, m_param.nodename.y);
	m_background->setPos(m_param.background.x, m_param.background.y);
	m_once->setPos(m_param.once.x, m_param.once.y);
	m_mute->setPos(m_param.mute.x, m_param.mute.y);
	m_view->setPos(m_param.view.x, m_param.view.y);
	m_genshin->setPos(m_param.genshin.x, m_param.genshin.y);
	*/
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