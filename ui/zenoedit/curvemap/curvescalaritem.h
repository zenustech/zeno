#ifndef __CURVE_SCALER_ITEM_H__
#define __CURVE_SCALER_ITEM_H__

#include <zenoui/model/modeldata.h>

class ZCurveMapView;

class CurveScalarItem : public QGraphicsObject
{
	typedef QGraphicsObject _base;
public:
	CurveScalarItem(bool bHorizontal, ZCurveMapView* pView, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

public slots:
	void update();

private:
	void resetPosition();

	const qreal sz = 24.;
	ZCurveMapView* m_view;
	bool m_bHorizontal;
};

#endif