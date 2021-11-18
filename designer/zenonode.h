#ifndef __ZENONODE_H__
#define __ZENONODE_H__

#include "renderparam.h"
#include "resizerectitem.h"

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

	ResizableRectItem* m_holder_nodename;
	ResizableRectItem* m_holder_status;
	ResizableRectItem* m_holder_control;
	ResizableRectItem* m_holder_display;
	ResizableRectItem* m_holder_header_backboard;

	ResizableRectItem* m_holder_topleftsocket;
	ResizableRectItem* m_holder_bottomleftsocket;
	ResizableRectItem* m_holder_toprightsocket;
	ResizableRectItem* m_holder_bottomrightsocket;

	ResizableRectItem* m_holder_body_backboard;

	NodeParam m_param;
};

#endif