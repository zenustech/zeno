#ifndef __CURVE_SCALER_ITEM_H__
#define __CURVE_SCALER_ITEM_H__

#include <zenoui/model/modeldata.h>

class CurveMapView;

class CurveScalarItem;

class CurveSliderItem : public QGraphicsObject
{
	typedef QGraphicsObject _base;
public:
	CurveSliderItem(CurveScalarItem* parent = nullptr);
	QRectF boundingRect() const override;
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    void resetPosition();

protected:
	QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

private:
	qreal clipPos(qreal x);
	qreal pos2val(qreal val);

	const qreal width = 38;
	const qreal height = 24;
	qreal m_value;
	CurveScalarItem* m_scalar;
	QGraphicsLineItem* m_line;
	qreal m_yoffset;
};


class CurveScalarItem : public QGraphicsObject
{
	Q_OBJECT
	typedef QGraphicsObject _base;
public:
	CurveScalarItem(bool bHorizontal, bool bFrame, CurveMapView* pView, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
    CURVE_RANGE range() const;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    int nFrames() const;

signals:
	void frameChanged(qreal);

public slots:
	void update();

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

private:
	void resetPosition();

	const qreal sz = 24.;
	CurveMapView* m_view;
	bool m_bHorizontal;
	bool m_bFrame;		//time frame scalar.
	int m_nFrames;
	CurveSliderItem* m_slider;
};

#endif