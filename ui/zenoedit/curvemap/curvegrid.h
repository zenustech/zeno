#ifndef __CURVE_GRID_H__
#define __CURVE_GRID_H__

#include <QtWidgets>

class CurveMapView;
class CurveNodeItem;
class CurvePathItem;

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
    bool isFuncCurve() const;
    int nodeCount() const;
	int indexOf(CurveNodeItem* pItem) const;
    QPointF nodePos(int i) const;
    CurveNodeItem *nodeItem(int i) const;

public slots:
    void onNodeGeometryChanged();
	void onNodeDeleted();
    void onPathClicked(const QPointF& pos);

private:
	QColor m_clrGrid, m_clrBg;
	CurveMapView* m_view;
	QRectF m_initRc;
	QTransform m_transform;
	QTransform m_invTrans;
	QVector<CurveNodeItem*> m_vecNodes;
	QVector<CurvePathItem*> m_vecCurves;
	bool m_bFCurve;			//function curve.
};


#endif