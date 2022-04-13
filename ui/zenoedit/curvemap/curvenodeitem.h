#ifndef __CURVENODE_ITEM_H__
#define __CURVENODE_ITEM_H__

#include <QtWidgets>
#include <QtSvg/QGraphicsSvgItem>
#include "curvegrid.h"

class CurveNodeItem;
class CurveMapView;

class CurvePathItem : public QObject
					, public QGraphicsPathItem
{
	Q_OBJECT
public:
    CurvePathItem(QGraphicsItem* parent = nullptr);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

signals:
    void clicked(QPointF);
};

class CurveHandlerItem : public QGraphicsObject
{
    Q_OBJECT
	typedef QGraphicsObject _base;
public:
	CurveHandlerItem(CurveNodeItem* pNode, const QPointF& pos, QGraphicsItem* parent = nullptr);
	~CurveHandlerItem();
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);
	QRectF boundingRect(void) const;
	void setOtherHandle(CurveHandlerItem* other);
    bool isMouseEventTriggered();
    void setUpdateNotify(bool bNotify);
    void toggle(bool bToggle);

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
	CurveNodeItem(CurveMapView* pView, const QPointF& nodePos, CurveGrid* parentItem = nullptr);
	void initHandles(const QPointF& leftHandle, const QPointF& rightHandle);
    void onHandleUpdate(CurveHandlerItem* pItem);
	QRectF boundingRect(void) const;
    void toggle(bool bChecked);
    CurveHandlerItem* leftHandle() const;
    CurveHandlerItem* rightHandle() const;
    QPointF leftHandlePos() const;
    QPointF rightHandlePos() const;
    CurveGrid* grid() const;
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);

signals:
	void geometryChanged();
	void deleteTriggered();

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
	void keyPressEvent(QKeyEvent* event);

private:
	CurveHandlerItem* m_left;
	CurveHandlerItem* m_right;
	CurveMapView* m_view;
	CurveGrid* m_grid;
	bool m_bToggle;
};


#endif