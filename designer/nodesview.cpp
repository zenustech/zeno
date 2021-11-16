#include "framework.h"
#include "nodesview.h"
#include "nodescene.h"


NodesView::NodesView(QWidget* parent)
	: QGraphicsView(parent)
	, m_scene(new NodeScene)
	, m_gridX(142)
	, m_gridY(76)
{
	setScene(m_scene);
	setBackgroundBrush(QBrush(QColor(30, 34, 36), Qt::SolidPattern));
}

void NodesView::initSkin(const QString& fn)
{
	m_scene->initSkin(fn);
}

void NodesView::initNode()
{
	m_scene->initNode();
}

QSize NodesView::sizeHint() const
{
	return QGraphicsView::sizeHint();
}

void NodesView::mousePressEvent(QMouseEvent* event)
{
	QPoint pos = event->pos();
	QTransform transform = this->transform();
	QPointF scenePos = mapToScene(pos);
	//scrollContentsBy(200, 200);
	QGraphicsView::mousePressEvent(event);
}