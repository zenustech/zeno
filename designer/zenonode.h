#ifndef __ZENONODE_H__
#define __ZENONODE_H__

#include "renderparam.h"

class ResizableComponentItem : public QGraphicsObject
{
	Q_OBJECT
public:
	ResizableComponentItem(int x, int y, int w, int h, QGraphicsItem* parent = nullptr);

	virtual QRectF boundingRect() const override;
	virtual QPainterPath shape() const override;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

private:
	const int dragW = 6;
	const int dragH = 8;
	const int borderW = 3;

	QGraphicsRectItem* m_borderItem;
	QGraphicsRectItem* m_ltcorner;
	QGraphicsRectItem* m_rtcorner;
	QGraphicsRectItem* m_lbcorner;
	QGraphicsRectItem* m_rbcorner;
};

class ZenoNode : public QGraphicsObject
{
	Q_OBJECT
public:
	ZenoNode(QGraphicsItem* parent = nullptr);
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