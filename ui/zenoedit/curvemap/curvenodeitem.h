#ifndef __CURVENODE_ITEM_H__
#define __CURVENODE_ITEM_H__

#include <QtWidgets>
#include <QtSvg/QGraphicsSvgItem>
#include "curvegrid.h"

class CurveNodeItem;
class CurveMapView;

class CurveHandlerItem : public QGraphicsRectItem
{
	typedef QGraphicsRectItem _base;
public:
	CurveHandlerItem(CurveNodeItem* pNode, const QModelIndex& idx, const QPointF& pos, QGraphicsItem* parent = nullptr);
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);
	QModelIndex index() const { return m_index; }
	void setOtherHandleIdx(const QModelIndex& idx);
	void updateStatus();

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
	const int sz = 6;
	QPersistentModelIndex m_index;
	QPersistentModelIndex m_nodeIdx;
	QPersistentModelIndex m_otherIdx;
	QGraphicsLineItem* m_line;
	CurveNodeItem* m_node;
};

class CurveNodeItem : public QGraphicsObject
{
	Q_OBJECT
	typedef QGraphicsObject _base;
public:
	CurveNodeItem(CurveMapView* pView, const QPointF& nodePos, QGraphicsItem* parentItem = nullptr);
	void initHandles(const MODEL_PACK& pack, const QModelIndex& idx, const QPointF& leftHandle, const QPointF& rightHandle);
	void updateStatus();
	void updateHandleStatus(const QString& objId);
	void onHandlerChanged(CurveHandlerItem* pHandler);
	QPointF logicPos() const;
	QRectF boundingRect(void) const;
	QModelIndex index() const { return m_index; }
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
	QPointF m_logicPos;
	QPersistentModelIndex m_index;
	CurveHandlerItem* m_left;
	CurveHandlerItem* m_right;
	CurveMapView* m_view;
	bool m_bToggle;
};


#endif