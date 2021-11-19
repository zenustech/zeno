#ifndef __ZENONODE_H__
#define __ZENONODE_H__

#include "renderparam.h"
#include "resizableitemimpl.h"

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

	ResizableItemImpl* m_holder_nodename;
	ResizableItemImpl* m_holder_status;
	ResizableItemImpl* m_holder_control;
	ResizableItemImpl* m_holder_display;
	ResizableItemImpl* m_holder_header_backboard;

	ResizableItemImpl* m_holder_topleftsocket;
	ResizableItemImpl* m_holder_bottomleftsocket;
	ResizableItemImpl* m_holder_toprightsocket;
	ResizableItemImpl* m_holder_bottomrightsocket;

	ResizableItemImpl* m_holder_body_backboard;

	NodeParam m_param;
};

#endif