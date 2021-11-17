#include "framework.h"
#include "zenonode.h"
#include "nodescene.h"
#include <rapidjson/document.h>

using namespace rapidjson;

DragPointItem::DragPointItem(const QRectF& rect, ResizableComponentItem* parent)
	: QGraphicsRectItem(rect, parent)
	, m_parent(parent)
{
	setFlags(ItemIsMovable | ItemSendsGeometryChanges);
}

void DragPointItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	//m_parent->setFlag(ItemIsMovable, false);
	QGraphicsRectItem::mouseMoveEvent(event);
	//m_parent->setFlag(ItemIsMovable, true);
	//m_parent->resizeByDragPoint(this);
}

void DragPointItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget /* = nullptr */)
{
	QGraphicsRectItem::paint(painter, option, widget);
	//painter->setBrush(brush());
	//QRectF r = rect();
	//r.setSize(r.size() - mRadius * QSizeF(1, 1));
	//r.translate(mRadius * QPointF(1, 1) / 2);
	//painter->drawRect(r);
}


ResizableComponentItem::ResizableComponentItem(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_width(w)
	, m_height(h)
	, m_ltcorner(nullptr)
	, m_rtcorner(nullptr)
	, m_lbcorner(nullptr)
	, m_rbcorner(nullptr)
{
	setPos(x, y);
	m_ltcorner = new DragPointItem(QRectF(0, 0, dragW, dragH), this);
	m_lbcorner = new DragPointItem(QRectF(0, 0, dragW, dragH), this);
	m_rtcorner = new DragPointItem(QRectF(0, 0, dragW, dragH), this);
	m_rbcorner = new DragPointItem(QRectF(0, 0, dragW, dragH), this);

	//m_ltcorner->installSceneEventFilter(this);
	//m_lbcorner->installSceneEventFilter(this);
	//m_rtcorner->installSceneEventFilter(this);
	//m_rbcorner->installSceneEventFilter(this);

	//setFlags(ItemIsMovable | ItemSendsGeometryChanges | ItemIsSelectable | ItemClipsToShape);
	setFlags(ItemIsMovable | ItemIsSelectable);
	_adjustItemsPos();
}

void ResizableComponentItem::_adjustItemsPos()
{
	m_ltcorner->setPos(-dragW / 2., -dragH / 2.);
	m_lbcorner->setPos(-dragW / 2., m_height - dragH / 2.);
	m_rtcorner->setPos(m_width - dragW / 2., -dragH / 2.);
	m_rbcorner->setPos(m_width - dragW / 2., m_height - dragH / 2.);
}

void ResizableComponentItem::resizeByDragPoint(DragPointItem* item)
{
	if (m_ltcorner == item)
	{
		QPointF topLeft = m_ltcorner->sceneBoundingRect().topLeft();
		QPointF bottomRight = m_rbcorner->sceneBoundingRect().topLeft();

		//m_lbcorner->setPos(mapFromScene(QPointF(topLeft.x(), bottomRight.y())));
		//m_rtcorner->setPos(mapFromScene(bottomRight.x(), topLeft.y()));

		topLeft = m_ltcorner->pos();
		bottomRight = m_rbcorner->pos();
		m_lbcorner->setPos(QPointF(topLeft.x(), bottomRight.y()));
		m_rtcorner->setPos(QPointF(bottomRight.x(), topLeft.y()));
	}
	else if (m_lbcorner == item)
	{

	}
	else if (m_rtcorner == item)
	{

	}
	else if (m_rbcorner == item)
	{

	}
	update();
}

bool ResizableComponentItem::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
	if (event->type() == QEvent::GraphicsSceneMouseMove)
	{
		QGraphicsSceneMouseEvent* mouseEvent = static_cast<QGraphicsSceneMouseEvent*>(event);
		if (watched == m_ltcorner)
		{
			QPointF pos = mouseEvent->pos();
			QPointF topLeft = pos;// m_ltcorner->pos() + QPointF(dragW / 2, dragH / 2);
			QPointF bottomRight = QPointF(m_width - 1, m_height - 1);
			m_width = abs(bottomRight.x() - topLeft.x() + 1);
			m_height = abs(bottomRight.y() - topLeft.y() + 1);
			setPos(mapToScene(topLeft));
			_adjustItemsPos();
			//_adjustItemsPos();
			//
			update();
			return true;
		}
		else if (watched == m_lbcorner)
		{
			int j;
			j = 0;
		}
		else if (watched == m_rtcorner)
		{
			int j;
			j = 0;
		}
		else if (watched == m_rbcorner)
		{
			int j;
			j = 0;
		}
	}
	else if (event->type() == QEvent::GraphicsSceneMouseRelease)
	{

	}
	return _base::sceneEventFilter(watched, event);
}

void ResizableComponentItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	QPointF scenePos = this->scenePos();
	scenePos = event->scenePos();
	_base::mousePressEvent(event);
}

void ResizableComponentItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	QPointF pos = event->pos();
	QPointF scenePos = event->scenePos();
	QList<QGraphicsItem*> items = scene()->items(scenePos);
	if (!items.isEmpty())
	{
		if (items[0] == m_ltcorner)
		{
			//m_width -= pos.x();
			//m_height -= pos.y();

			//QPointF topLeft = scenePos;
			//QPointF bottomRight = mapToScene(QPointF(m_width - 1, m_height - 1));
			//m_width = abs(bottomRight.x() - topLeft.x() + 1);
			//m_height = abs(bottomRight.y() - topLeft.y() + 1);
			//setPos(scenePos);
			//_adjustItemsPos();
			//update();
			return;
		}
	}
	_base::mouseMoveEvent(event);
}

QRectF ResizableComponentItem::boundingRect() const
{
	QPointF lt = m_ltcorner->pos();
	QPointF rb = m_rbcorner->pos() + m_rbcorner->boundingRect().bottomRight();

	QRectF rcLT = mapRectFromScene(m_ltcorner->sceneBoundingRect());
	QRectF rcRB = mapRectFromScene(m_rbcorner->sceneBoundingRect());
	//return QRectF(lt, rb);
	QRectF rcc = childrenBoundingRect();
	return rcc;
	//return QRectF(0, 0, m_width, m_height).adjusted(-dragW / 2., -dragH / 2, dragW / 2, dragH / 2);
}

QPainterPath ResizableComponentItem::shape() const
{
	return _base::shape();
}

void ResizableComponentItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	if (option->state & (QStyle::State_Selected | QStyle::State_HasFocus))
	{
		QPen pen(QColor(21, 152, 255), borderW);
		pen.setJoinStyle(Qt::MiterJoin);
		QBrush brush(QColor(255, 255, 255));
		
		m_ltcorner->setPen(pen);
		m_ltcorner->setBrush(brush);

		m_lbcorner->setPen(pen);
		m_lbcorner->setBrush(brush);

		m_rtcorner->setPen(pen);
		m_rtcorner->setBrush(brush);

		m_rbcorner->setPen(pen);
		m_rbcorner->setBrush(brush);

		m_ltcorner->show();
		m_lbcorner->show();
		m_rtcorner->show();
		m_rbcorner->show();

		painter->setPen(pen);
		painter->setBrush(Qt::NoBrush);
		QPointF lt = mapFromScene(m_ltcorner->sceneBoundingRect().center());
		QPointF rb = mapFromScene(m_rbcorner->sceneBoundingRect().center());
		painter->drawRect(QRectF(lt, rb));
		//painter->drawRect(QRectF(0, 0, m_width, m_height));
	}
	else
	{
		QPen pen(QColor(0, 0, 0), borderW);
		pen.setJoinStyle(Qt::MiterJoin);

		m_ltcorner->hide();
		m_lbcorner->hide();
		m_rtcorner->hide();
		m_rbcorner->hide();

		painter->setPen(pen);
		painter->setBrush(Qt::NoBrush);
		QPointF lt = mapFromScene(m_ltcorner->sceneBoundingRect().center());
		QPointF rb = mapFromScene(m_rbcorner->sceneBoundingRect().center());
		painter->drawRect(QRectF(lt, rb));
		//painter->drawRect(QRectF(0, 0, m_width, m_height));
	}
}


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
	m_holder_nodename = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);
	QPointF pos = m_holder_nodename->scenePos();
	int j;
	j = 0;
	/*
	comp = m_param.header.status;
	m_holder_status = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.header.backborad;
	m_holder_header_backboard = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);
	m_holder_header_backboard->setZValue(-10);

	comp = m_param.header.display;
	m_holder_display = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.header.control;
	m_holder_control = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);

	comp = m_param.body.backboard;
	m_holder_body_backboard = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);
	m_holder_body_backboard->setZValue(-10);

	comp = m_param.body.leftTop;
	m_holder_topleftsocket = new ResizableComponentItem(comp.x, comp.y, comp.w, comp.h, this);
	*/

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