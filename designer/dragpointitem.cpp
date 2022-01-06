#include "framework.h"
#include "dragpointitem.h"

DragPointItem::DragPointItem(NodeScene::DRAG_ITEM dragObj, NodeScene* pScene, int w, int h)
	: _base(0, 0, w, h)
	, m_pScene(pScene)
	, m_dragObj(dragObj)
{
	setFlags(ItemIsMovable | ItemSendsGeometryChanges | ItemSendsScenePositionChanges);
}

void DragPointItem::setWatchedObject(QGraphicsItem* pObject)
{

}

void DragPointItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	_base::mousePressEvent(event);
}

void DragPointItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	_base::mouseMoveEvent(event);
	m_pScene->updateDragPoints(this, m_dragObj);
}

void DragPointItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	_base::mouseReleaseEvent(event);
}
