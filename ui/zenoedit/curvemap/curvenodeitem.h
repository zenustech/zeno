#ifndef __CURVENODE_ITEM_H__
#define __CURVENODE_ITEM_H__

#include <QtWidgets>
#include <QtSvg/QGraphicsSvgItem>
#include "curvegrid.h"
#include "curveutil.h"

class CurveNodeItem;
class CurveMapView;
class CurvesItem;

using namespace curve_util;

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
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) override;
	QRectF boundingRect(void) const override;
	void setOtherHandle(CurveHandlerItem* other);
    bool isMouseEventTriggered();
    void setUpdateNotify(bool bNotify);
    void toggle(bool bToggle);
	enum{ Type = curve_util::CURVE_HANDLE };
    int type() const override;
	CurveNodeItem* nodeItem() const;

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
	CurveNodeItem(const QModelIndex& idx, CurveMapView* pView, const QPointF& nodePos, CurveGrid* parentItem, CurvesItem* curve);
	void initHandles(const QPointF& leftHandle, const QPointF& rightHandle);
    void onHandleUpdate(CurveHandlerItem* pItem);
	QRectF boundingRect(void) const override;
    void toggle(bool bChecked);
    bool isToggled() const;
    CurveHandlerItem* leftHandle() const;
    CurveHandlerItem* rightHandle() const;
    QPointF leftHandlePos() const;
    QPointF rightHandlePos() const;
    CurveGrid* grid() const;
	CurvesItem* curves() const;

	enum{ Type = curve_util::CURVE_NODE };
    int type() const override;
    HANDLE_TYPE hdlType() const;
    void setHdlType(HANDLE_TYPE type);

	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) override;

signals:
	void geometryChanged();
	void deleteTriggered();

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
	void keyPressEvent(QKeyEvent* event) override;

private:
	CurveHandlerItem* m_left;
	CurveHandlerItem* m_right;
	CurveMapView* m_view;
	CurvesItem* m_curve;
	CurveGrid* m_grid;
	HANDLE_TYPE m_type;
	QPersistentModelIndex m_index;
	bool m_bToggle;
};


#endif
