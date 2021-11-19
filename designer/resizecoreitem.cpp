#include "resizecoreitem.h"

ResizableCoreItem::ResizableCoreItem(QGraphicsItem* parent)
	: QGraphicsItem(parent)
{

}

void ResizableCoreItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}


/////////////////////////////////////////////////////////////////////////////////////
ResizablePixmapItem::ResizablePixmapItem(const QPixmap& pixmap, QGraphicsItem* parent)
	: ResizableCoreItem(parent)
	, m_original(pixmap)
	, m_pixmapitem(new QGraphicsPixmapItem(pixmap, this))
{
}

QRectF ResizablePixmapItem::boundingRect() const
{
	return m_pixmapitem->boundingRect();
}

void ResizablePixmapItem::resize(QSizeF sz)
{
	m_pixmapitem->setPixmap(m_original.scaledToWidth(sz.width()));
}


//////////////////////////////////////////////////////////////////////////////////////
ResizableRectItem::ResizableRectItem(QRectF rc, QGraphicsItem* parent)
	: m_rectItem(new QGraphicsRectItem(rc, parent))
{
	QPen pen(QColor(255, 0, 0), 1);
	pen.setJoinStyle(Qt::MiterJoin);
	m_rectItem->setPen(pen);
	m_rectItem->setBrush(QColor(142, 101, 101));
}

QRectF ResizableRectItem::boundingRect() const
{
	return m_rectItem->boundingRect();
}

void ResizableRectItem::resize(QSizeF sz)
{
	QPointF topLeft = m_rectItem->rect().topLeft();
	m_rectItem->setRect(QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height()));
}


///////////////////////////////////////////////////////////////////////////////////////
ResizableEclipseItem::ResizableEclipseItem(const QRectF& rect, QGraphicsItem* parent)
	: ResizableCoreItem(parent)
	, m_ellipseItem(new QGraphicsEllipseItem(rect, this))
{
	QPen pen(QColor(255, 0, 0), 1);
	pen.setJoinStyle(Qt::MiterJoin);
	m_ellipseItem->setPen(pen);
	m_ellipseItem->setBrush(QColor(142, 101, 101));
}

QRectF ResizableEclipseItem::boundingRect() const
{
	return m_ellipseItem->boundingRect();
}

void ResizableEclipseItem::resize(QSizeF sz)
{
	QPointF topLeft = m_ellipseItem->rect().topLeft();
	m_ellipseItem->setRect(QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height()));
}


////////////////////////////////////////////////////////////////////////////////////////
ResizableTextItem::ResizableTextItem(const QString& text, QGraphicsItem* parent)
	: m_pTextItem(new QGraphicsTextItem(text, this))
{
}

QRectF ResizableTextItem::boundingRect() const
{
	return m_pTextItem->boundingRect();
}

void ResizableTextItem::resize(QSizeF sz)
{
}