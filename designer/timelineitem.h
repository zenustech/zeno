#ifndef __TIMELINE_ITEM_H__
#define __TIMELINE_ITEM_H__

class NodeScene;

class TimelineItemTemp : public QGraphicsRectItem
{
public:
	TimelineItemTemp(QRectF rcView, QGraphicsItem *parent = nullptr);

protected:
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
};

class TimelineItem : public QGraphicsObject
{
	Q_OBJECT
public:
	TimelineItem(NodeScene* pScene, bool bHorizontal, QRectF rcView, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

public slots:
	void resetPosition();

private:
	int _getframes();

	qreal m_left, m_top, m_right, m_bottom, m_from, m_to;
	const int sz = 24;
	QRectF m_view;
	NodeScene* m_pScene;
	bool m_bHorizontal;
};

#endif