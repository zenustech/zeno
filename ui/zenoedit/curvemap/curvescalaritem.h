#ifndef __CURVE_SCALER_ITEM_H__
#define __CURVE_SCALER_ITEM_H__

#include <zenoui/model/modeldata.h>

class CurveMapView;

class CurveScalarItem : public QGraphicsObject
{
	typedef QGraphicsObject _base;
public:
	CurveScalarItem(bool bHorizontal, CurveMapView* pView, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

public slots:
	void update();

private:
	void resetPosition();

	const qreal sz = 24.;
	CurveMapView* m_view;
	bool m_bHorizontal;
};

#endif