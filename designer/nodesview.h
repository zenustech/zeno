#ifndef __NODESVIEW_H__
#define __NODESVIEW_H__

class NodeScene;
class NodesView : public QGraphicsView
{
	Q_OBJECT
public:
	NodesView(QWidget* parent = nullptr);
	QSize sizeHint() const override;
	NodeScene* scene() const { return m_scene; }
	void initSkin(const QString& fn);
	void initNode();

protected:
	void mousePressEvent(QMouseEvent* event);
	void mouseMoveEvent(QMouseEvent* event);
	void mouseReleaseEvent(QMouseEvent* event);
	void wheelEvent(QWheelEvent* event);
	void paintEvent(QPaintEvent* event);
	void drawForeground(QPainter* painter, const QRectF& rect);

signals:
	void zoomed(qreal);
	void viewChanged(qreal);

private:
	void gentle_zoom(qreal factor);
	void set_modifiers(Qt::KeyboardModifiers modifiers);
	void zoomIn();
	void zoomOut();
	void resetTransform();
	qreal _factorStep(qreal factor);

	QPointF target_scene_pos, target_viewport_pos, m_startPos;

	int m_gridX;
	int m_gridY;
	qreal m_factor;
	const double m_factor_step = 0.1;
	bool m_dragMove;
	NodeScene* m_scene;
	Qt::KeyboardModifiers _modifiers;
};


#endif