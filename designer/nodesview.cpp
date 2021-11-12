#include "framework.h"
#include "nodesview.h"
#include "nodescene.h"


NodesView::NodesView(QWidget* parent)
	: QGraphicsView(parent)
	, m_scene(new NodeScene)
{
	setScene(m_scene);
	setBackgroundBrush(QBrush(QColor(51,59,62), Qt::SolidPattern));
}