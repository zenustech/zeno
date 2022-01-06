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
	void updateScalar(qreal factor);

private:
	QRectF m_view;
	qreal m_left, m_top, m_right, m_bottom, m_from, m_to;
	NodeScene* m_pScene;
	const int sz = 24;
	int m_nframes;
	qreal m_factor;
	bool m_bHorizontal;
};

#endif