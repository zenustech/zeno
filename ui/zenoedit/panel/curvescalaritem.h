#ifndef __CURVE_SCALER_ITEM_H__
#define __CURVE_SCALER_ITEM_H__

#include <zenoui/model/modeldata.h>

class ZCurveMapView;

class CurveScalarItem : public QGraphicsObject
{
public:
	CurveScalarItem(bool bHorizontal, ZCurveMapView* pView, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

public slots:
	void resetPosition(QGraphicsView* pView);
	void updateScalar(QGraphicsView* pView, qreal factor, int nFrames);
	void onResizeView(QGraphicsView* pView);

private:
	QRectF m_rect;
	qreal m_from, m_to;
	CURVE_RANGE m_range;
	const qreal sz = 24.;
	int m_nframes;
	qreal m_factor;
	ZCurveMapView* m_view;
	bool m_bHorizontal;
};

#endif