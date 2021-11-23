#ifndef __ZENONODE_H__
#define __ZENONODE_H__

#include "renderparam.h"
#include "resizableitemimpl.h"
#include "common.h"

class NodeScene;

class ZenoNode : public QGraphicsObject
{
	Q_OBJECT
public:
	ZenoNode(NodeScene* pScene, QGraphicsItem* parent = nullptr);
	void initStyle(const NodeParam& param);
	void initModel(QStandardItemModel* pModel, QItemSelectionModel* selection);

	virtual QRectF boundingRect() const override;
	virtual QPainterPath shape() const override;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

public slots:
	void onSelectionChanged(const QItemSelection&, const QItemSelection&);

private:
	QStandardItem* createItemWithGVItem(ResizableItemImpl* gvItem, NODE_ID id, const QString& name, QItemSelectionModel* selection);

	QGraphicsPixmapItem* m_once;
	QGraphicsPixmapItem* m_prep;
	QGraphicsPixmapItem* m_mute;
	QGraphicsPixmapItem* m_view;

	QGraphicsPixmapItem* m_genshin;
	QGraphicsPixmapItem* m_background;
	QGraphicsTextItem* m_nodename;

	ResizableItemImpl* m_component_nodename;
	ResizableItemImpl* m_component_status;
	ResizableItemImpl* m_component_control;
	ResizableItemImpl* m_component_display;
	ResizableItemImpl* m_component_header_backboard;

	ResizableItemImpl* m_component_ltsocket;
	ResizableItemImpl* m_component_lbsocket;
	ResizableItemImpl* m_component_rtsocket;
	ResizableItemImpl* m_component_rbsocket;

	ResizableItemImpl* m_component_body_backboard;

	NodeParam m_param;
};

#endif