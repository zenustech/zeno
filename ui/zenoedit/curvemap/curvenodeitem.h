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
	CurveHandlerItem(CurveNodeItem* pNode, const QPointF& pos, QGraphicsItem* parent = nullptr);
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);
	void setOtherHandleIdx(const QModelIndex& idx);
	void setOtherHandle(CurveHandlerItem* other);
    bool isMouseEventTriggered();
    void setUpdateNotify(bool bNotify);

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
	const int sz = 6;
	QGraphicsLineItem* m_line;
	CurveNodeItem* m_node;
	CurveHandlerItem* m_other;
	bool m_bMouseTriggered;
	bool m_bNotify;
};

class CurveNodeItem : public QGraphicsObject
{
	Q_OBJECT
	typedef QGraphicsObject _base;
public:
	CurveNodeItem(CurveMapView* pView, const QPointF& nodePos, QGraphicsItem* parentItem = nullptr);
	void initHandles(const QPointF& leftHandle, const QPointF& rightHandle);
    void onHandleUpdate(CurveHandlerItem* pItem);
	QRectF boundingRect(void) const;
    void toggle(bool bChecked);
    QPointF leftHandlePos() const;
    QPointF rightHandlePos() const;
    void setLeftCurve(QGraphicsPathItem* leftCurve);
    void setRightCurve(QGraphicsPathItem* rightCurve);
    QGraphicsPathItem* leftCurve() const;
    QGraphicsPathItem* rightCurve() const;
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);

signals:
	void geometryChanged();

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
	CurveHandlerItem* m_left;
	CurveHandlerItem* m_right;
    QGraphicsPathItem* m_leftCurve;
    QGraphicsPathItem* m_rightCurve;
	CurveMapView* m_view;
	bool m_bToggle;
};


#endif