#ifndef __CURVE_GRID_H__
#define __CURVE_GRID_H__

#include <QtWidgets>

class ZCurveMapView;

class CurveGrid : public QGraphicsObject
{
	Q_OBJECT
public:
	CurveGrid(ZCurveMapView* pView, QGraphicsItem* parent = nullptr);
	void setColor(const QColor& clrGrid, const QColor& clrBackground);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

private:
	QColor m_clrGrid, m_clrBg;
	ZCurveMapView* m_view;
};


#endif