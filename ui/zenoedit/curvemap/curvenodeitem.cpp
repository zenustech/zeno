#include "curvenodeitem.h"


CurveHandlerItem::CurveHandlerItem(CurveNodeItem* pNode, const QPointF& pos, QGraphicsItem* parent)
	: QGraphicsRectItem(parent)
	, m_node(pNode)
{
	m_line = new QGraphicsLineItem(this);
	m_line->setPen(QPen(QColor(255,255,255), 2));
	setBrush(QColor(255, 0, 0));
	QPen pen;
	pen.setColor(QColor(255,255,255));
	pen.setWidth(2);
	setPen(pen);
}

QVariant CurveHandlerItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF nodePos = mapFromScene(m_node->scenePos());
		QPointF thisPos = boundingRect().center();
		m_line->setLine(QLineF(thisPos, nodePos));
		m_node->onHandlerChanged(this);
	}
	return value;
}


CurveNodeItem::CurveNodeItem(const QPointF& nodePos, const QPointF& leftHandle, const QPointF& rightHandle, QGraphicsItem* parentItem)
	: QGraphicsSvgItem(":/icons/collaspe.svg", parentItem)
	, m_left(nullptr)
	, m_right(nullptr)
{
	m_left = new CurveHandlerItem(this, leftHandle, this);
	m_right = new CurveHandlerItem(this, rightHandle, this);
	m_left->hide();
	m_right->hide();
}

QVariant CurveNodeItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		m_left->setVisible(value.toBool());
		m_right->setVisible(value.toBool());
	}
	return value;
}

void CurveNodeItem::onHandlerChanged(CurveHandlerItem* pHandler)
{
	if (pHandler == m_left)
	{

	}
	else
	{

	}
}