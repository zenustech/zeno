#ifndef __CURVE_GRID_H__
#define __CURVE_GRID_H__

#include <QtWidgets>

class CurveMapView;

class CurveGrid : public QGraphicsObject
{
	Q_OBJECT
public:
	CurveGrid(CurveMapView* pView, QGraphicsItem* parent = nullptr);
	void setColor(const QColor& clrGrid, const QColor& clrBackground);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

private:
	QColor m_clrGrid, m_clrBg;
	CurveMapView* m_view;
};


#endif