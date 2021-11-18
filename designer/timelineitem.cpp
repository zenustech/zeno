#include "framework.h"
#include "timelineitem.h"


TimelineItem::TimelineItem(QGraphicsItem* parent)
	: QGraphicsRectItem(0, 0, 1000, 20, parent)
{
	setPen(Qt::NoPen);
	setBrush(QColor(51, 51, 51));
}

void TimelineItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	QGraphicsRectItem::paint(painter, option, widget);
}