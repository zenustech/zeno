#ifndef __CURVEMAP_VIEW_H__
#define __CURVEMAP_VIEW_H__

#include <QtWidgets>
#include <zenoui/model/modeldata.h>

class CurveGrid;
class CurveScalarItem;
class CurveNodeItem;

class CurveMapView : public QGraphicsView
{
	Q_OBJECT
public:
	CurveMapView(QWidget* parent = nullptr);
	~CurveMapView();
	void init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers);
	CURVE_RANGE range() const { return m_range; }
	int frames(bool bHorizontal) const;
    bool isSmoothCurve() const;
	QMargins margins() const { return m_gridMargins; }
	qreal factor() const { return m_factor; }
	QRectF gridBoundingRect() const;
	QPointF mapLogicToScene(const QPointF& logicPos);
	QPointF mapSceneToLogic(const QPointF& scenePos);
	QPointF mapOffsetToScene(const QPointF& offset);

protected:
	void wheelEvent(QWheelEvent* event);
	void drawBackground(QPainter* painter, const QRectF& rect) override;
	void mousePressEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void resizeEvent(QResizeEvent* event);

private:
	void gentle_zoom(qreal factor);
	void set_modifiers(Qt::KeyboardModifiers modifiers);
	void resetTransform();

	QPointF target_scene_pos, target_viewport_pos, m_startPos;
	QPoint m_mousePos;
	QPoint _last_mouse_pos;
	qreal m_factor;
	QString m_path;
	CURVE_RANGE m_range;
	QMargins m_gridMargins;
	QRectF m_fixedSceneRect;
	CurveGrid* m_grid;
	CurveScalarItem* m_pHScalar;
	CurveScalarItem* m_pVScalar;
	QVector<CurveNodeItem*> m_nodes;
	QVector<QGraphicsPathItem*> m_curves;
	const double m_factor_step = 0.1;
	Qt::KeyboardModifiers _modifiers;
	bool m_dragMove;
	bool m_bSmoothCurve;
};

#endif