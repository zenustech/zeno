

#ifndef __DRAGPOINT_ITEM_H__
#define __DRAGPOINT_ITEM_H__

#include "nodescene.h"

class DragPointItem : public QGraphicsRectItem
{
	typedef QGraphicsRectItem _base;
public:
	DragPointItem(NodeScene::DRAG_ITEM dragObj, NodeScene* pScene, int w, int h);
	void setWatchedObject(QGraphicsItem* pObject);

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

private:
	NodeScene* m_pScene;
	NodeScene::DRAG_ITEM m_dragObj;
};

#endif