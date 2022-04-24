#ifndef __CURVE_GRID_H__
#define __CURVE_GRID_H__

#include <QtWidgets>

class CurveMapView;
class CurveNodeItem;
class CurvePathItem;
class CurvesItem;
class CurveModel;

class CurveGrid : public QGraphicsObject
{
	Q_OBJECT
public:
	CurveGrid(CurveMapView* pView, const QRectF& rc, QGraphicsItem* parent = nullptr);
	void setColor(const QColor& clrGrid, const QColor& clrBackground);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void initTransform();
	void addCurve(CurveModel* model);
    bool isFuncCurve() const;
    CurveMapView* view() const { return m_view; }
    QPointF logicToScene(QPointF logicPos);
    QPointF sceneToLogic(QPointF scenePos);

private:
	QColor m_clrGrid, m_clrBg;
	QRectF m_initRc;
	QTransform m_transform;
	QTransform m_invTrans;
	CurveMapView* m_view;
	QMap<QString, CurvesItem*> m_curves;
	bool m_bFCurve;			//function curve.
};


#endif
