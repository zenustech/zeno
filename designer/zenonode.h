#ifndef __ZENONODE_H__
#define __ZENONODE_H__

#include "renderparam.h"

class ResizableComponentItem;
class DragPointItem : public QGraphicsRectItem
{
public:
	DragPointItem(const QRectF& rect, ResizableComponentItem* parent = nullptr);
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);

protected:
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

private:
	ResizableComponentItem* m_parent;
};

class ResizableComponentItem : public QGraphicsObject
{
	Q_OBJECT
	typedef QGraphicsObject _base;
public:
	ResizableComponentItem(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent = nullptr);

	virtual QRectF boundingRect() const override;
	virtual QPainterPath shape() const override;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
	void resizeByDragPoint(DragPointItem* item);

protected:
	bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

private:
	void _adjustItemsPos();

	const qreal dragW = 8.;
	const qreal dragH = 8.;
	const qreal borderW = 2.;

	qreal m_width, m_height;

	QGraphicsRectItem* m_ltcorner;
	QGraphicsRectItem* m_rtcorner;
	QGraphicsRectItem* m_lbcorner;
	QGraphicsRectItem* m_rbcorner;
};

class NodeScene;

class ZenoNode : public QGraphicsObject
{
	Q_OBJECT
public:
	ZenoNode(NodeScene* pScene, QGraphicsItem* parent = nullptr);
	void initStyle(const NodeParam& param);

	virtual QRectF boundingRect() const override;
	virtual QPainterPath shape() const override;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

private:
	QGraphicsPixmapItem* m_once;
	QGraphicsPixmapItem* m_prep;
	QGraphicsPixmapItem* m_mute;
	QGraphicsPixmapItem* m_view;

	QGraphicsPixmapItem* m_genshin;
	QGraphicsPixmapItem* m_background;
	QGraphicsTextItem* m_nodename;

	ResizableComponentItem* m_holder_nodename;
	ResizableComponentItem* m_holder_status;
	ResizableComponentItem* m_holder_control;
	ResizableComponentItem* m_holder_display;
	ResizableComponentItem* m_holder_header_backboard;
	ResizableComponentItem* m_holder_topleftsocket;
	ResizableComponentItem* m_holder_body_backboard;

	NodeParam m_param;
};

#endif