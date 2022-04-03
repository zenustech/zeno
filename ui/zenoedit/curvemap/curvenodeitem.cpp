#include "curvenodeitem.h"
#include "curvemapview.h"


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

	QPointF wtf = mapFromScene(pos);
	setPos(pos);

	//setPos(QPointF(-50, 50));

	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
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


CurveNodeItem::CurveNodeItem(CurveMapView* pView, const QPointF& nodePos, const QPointF& leftHandle, const QPointF& rightHandle, QGraphicsItem* parentItem)
	: QGraphicsSvgItem(":/icons/collaspe.svg", parentItem)
	, m_left(nullptr)
	, m_right(nullptr)
	, m_view(pView)
{
	QPointF pos = m_view->mapOffsetToScene(leftHandle);
	m_left = new CurveHandlerItem(this, pos, this);

	pos = m_view->mapOffsetToScene(rightHandle);
	m_right = new CurveHandlerItem(this, pos, this);

	m_left->hide();
	m_right->hide();
	m_logicPos = m_view->mapSceneToLogic(nodePos);
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
}

QVariant CurveNodeItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		m_left->setVisible(value.toBool());
		QPointF scenePos = m_left->scenePos();
		m_right->setVisible(value.toBool());
		scenePos = m_right->scenePos();
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF phyPos = scenePos();
		m_logicPos = m_view->mapSceneToLogic(phyPos);
	}
	return value;
}

void CurveNodeItem::updatePos()
{
	QPointF scenePos = m_view->mapLogicToScene(m_logicPos);
	setPos(scenePos);
}

void CurveNodeItem::updateScale()
{

}

QPointF CurveNodeItem::logicPos() const
{
	return m_logicPos;
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