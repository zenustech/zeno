#ifndef __CURVENODE_ITEM_H__
#define __CURVENODE_ITEM_H__

#include <QtWidgets>
#include <QtSvg/QGraphicsSvgItem>

class CurveNodeItem;

class CurveHandlerItem : public QObject
					   , public QGraphicsRectItem
{
	Q_OBJECT
public:
	CurveHandlerItem(CurveNodeItem* pNode, const QPointF& pos, QGraphicsItem* parent = nullptr);

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
	QGraphicsLineItem* m_line;
	CurveNodeItem* m_node;
};

class CurveNodeItem : public QGraphicsSvgItem
{
	Q_OBJECT
public:
	CurveNodeItem(const QPointF& nodePos, const QPointF& leftHandle, const QPointF& rightHandle, QGraphicsItem* parentItem = nullptr);
	void onHandlerChanged(CurveHandlerItem* pHandler);

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
	CurveHandlerItem* m_left;
	CurveHandlerItem* m_right;
};


#endif