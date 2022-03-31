#ifndef __CURVE_GRID_H__
#define __CURVE_GRID_H__

#include <QtWidgets>

class CurveGrid : public QGraphicsObject
{
	Q_OBJECT
public:
	CurveGrid(QGraphicsItem* parent = nullptr);
	void reset(const QRectF& rc);
	void setColor(const QColor& clrGrid, const QColor& clrBackground);
	void setFactor(const qreal& factor, int nFrames);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

private:
	qreal m_factor;
	int m_nFrames;
	QRectF m_sceneRect;
	QColor m_clrGrid, m_clrBg;
};


#endif