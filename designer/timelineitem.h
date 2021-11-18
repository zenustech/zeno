#ifndef __TIMELINE_ITEM_H__
#define __TIMELINE_ITEM_H__

class TimelineItem : public QGraphicsRectItem
{
public:
	TimelineItem(QGraphicsItem *parent = nullptr);

protected:
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
};

#endif