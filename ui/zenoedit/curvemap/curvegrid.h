#ifndef __CURVE_GRID_H__
#define __CURVE_GRID_H__

#include <QtWidgets>

class CurveMapView;
class CurveNodeItem;

struct MODEL_PACK
{
	QStandardItemModel* pModel;
	QItemSelectionModel* pSelection;
};

class CurveGrid : public QGraphicsObject
{
	Q_OBJECT
public:
	CurveGrid(CurveMapView* pView, const QRectF& rc, QGraphicsItem* parent = nullptr);
	void setColor(const QColor& clrGrid, const QColor& clrBackground);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
	void mousePressEvent(QGraphicsSceneMouseEvent* event);
	void initTransform();
	void initCurves(const QVector<QPointF>& pts, const QVector<QPointF>& handlers);

public slots:
    void onNodeGeometryChanged();

private:
	QColor m_clrGrid, m_clrBg;
	CurveMapView* m_view;
	QRectF m_initRc;
	QTransform m_transform;
	QTransform m_invTrans;
	QVector<CurveNodeItem*> m_vecNodes;
};


#endif