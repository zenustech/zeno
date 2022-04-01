#ifndef __ZCURVEMAP_EDITOR_H__
#define __ZCURVEMAP_EDITOR_H__

#include <zenoui/model/modeldata.h>
#include <QtWidgets>

class CurveScalarItem;
class CurveGrid;
class NodeGridItem;

class ZCurveMapView : public QGraphicsView
{
	Q_OBJECT
public:
	ZCurveMapView(QWidget* parent = nullptr);
	~ZCurveMapView();
	void init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers);
	CURVE_RANGE range() const;

protected:
	void wheelEvent(QWheelEvent* event);
	void drawBackground(QPainter* painter, const QRectF& rect) override;
	void mousePressEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void resizeEvent(QResizeEvent* event);

signals:
	void zoomed(qreal);
	void viewChanged(qreal);

private:
	void gentle_zoom(qreal factor);
	void set_modifiers(Qt::KeyboardModifiers modifiers);
	void resetTransform();
	void drawGrid(QPainter* painter, const QRectF& rect);
	int metrics(int factor) const;

	QPointF target_scene_pos, target_viewport_pos, m_startPos;
	QPoint m_mousePos;
	QPoint _last_mouse_pos;
	qreal m_factor;
	QString m_path;
	CURVE_RANGE m_range;
	CurveGrid* m_grid;
	CurveScalarItem* m_pHScalar;
	CurveScalarItem* m_pVScalar;
	const double m_factor_step = 0.1;
	Qt::KeyboardModifiers _modifiers;
	bool m_dragMove;
};

class ZCurveMapEditor : public QWidget
{
	Q_OBJECT
public:
	ZCurveMapEditor(QWidget* parent = nullptr);
	~ZCurveMapEditor();
	void init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers);

private:
	ZCurveMapView* m_view;
};


#endif