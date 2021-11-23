#include "framework.h"
#include "nodesview.h"
#include "nodescene.h"


NodesView::NodesView(QWidget* parent)
	: QGraphicsView(parent)
	, m_scene(new NodeScene(this))
	, m_gridX(142)
	, m_gridY(76)
	, _modifiers(Qt::ControlModifier)
	, m_factor(1.)
	, m_dragMove(false)
{
	QRectF rcView(QPointF(-2400, -3700), QPointF(3300, 4500));

	setScene(m_scene);
	setSceneRect(rcView);
	m_scene->initGrid();
	m_scene->initTimelines(rcView);
	setBackgroundBrush(QBrush(QColor(30, 34, 36), Qt::SolidPattern));
	setDragMode(QGraphicsView::NoDrag);
	setTransformationAnchor(QGraphicsView::NoAnchor);
	viewport()->installEventFilter(this);
	setMouseTracking(true);

	connect(m_scene, SIGNAL(changed(QList<QRectF>)), this, SLOT(update()));
	connect(this, SIGNAL(viewChanged(qreal)), m_scene, SLOT(onViewTransformChanged(qreal)));
}

void NodesView::initSkin(const QString& fn)
{
	m_scene->initSkin(fn);
}

void NodesView::initNode()
{
	m_scene->initNode();
}

QSize NodesView::sizeHint() const
{
	return QGraphicsView::sizeHint();
}

void NodesView::gentle_zoom(qreal factor)
{
	scale(factor, factor);
	centerOn(target_scene_pos);
	QPointF delta_viewport_pos = target_viewport_pos - 
		QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
	QPointF viewport_center = mapFromScene(target_scene_pos) - delta_viewport_pos;
	centerOn(mapToScene(viewport_center.toPoint()));

	qreal factor_i_want = transform().m11();
	emit zoomed(factor_i_want);
	emit viewChanged(m_factor);
}

qreal NodesView::_factorStep(qreal factor)
{
	if (factor < 2)
		return 0.1;
	else if (factor < 3)
		return 0.25;
	else if (factor < 4)
		return 0.4;
	else if (factor < 10)
		return 0.7;
	else if (factor < 20)
		return 1.0;
	else
		return 1.5;
}

void NodesView::zoomIn()
{
	m_factor = std::min(m_factor + _factorStep(m_factor), 32.0);
	qreal current_factor = transform().m11();
	qreal factor_complicate = m_factor / current_factor;
	gentle_zoom(factor_complicate);
}

void NodesView::zoomOut()
{
	m_factor = std::max(m_factor - _factorStep(m_factor), 0.4);
	qreal current_factor = transform().m11();
	qreal factor_complicate = m_factor / current_factor;
	gentle_zoom(factor_complicate);
}

void NodesView::set_modifiers(Qt::KeyboardModifiers modifiers)
{
	_modifiers = modifiers;
}

void NodesView::resetTransform()
{
	QGraphicsView::resetTransform();
	m_factor = 1.0;
}

void NodesView::mousePressEvent(QMouseEvent* event)
{
	if (event->type() == QEvent::MouseButtonPress &&
		QApplication::keyboardModifiers() == _modifiers)
	{
		m_dragMove = true;
		m_startPos = event->pos();
	}
	QGraphicsView::mousePressEvent(event);
}

void NodesView::mouseMoveEvent(QMouseEvent* mouse_event)
{
	QPointF delta = target_viewport_pos - mouse_event->pos();
	if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5)
	{
		target_viewport_pos = mouse_event->pos();
		target_scene_pos = mapToScene(mouse_event->pos());
	}
	if (m_dragMove)
	{
		QPointF delta = m_startPos - mouse_event->pos();
		QTransform transform = this->transform();
		qreal deltaX = delta.x() / transform.m11();
		qreal deltaY = delta.y() / transform.m22();
		translate(-deltaX, -deltaY);
		m_startPos = mouse_event->pos();
		emit viewChanged(m_factor);
		return;
	}
	QGraphicsView::mouseMoveEvent(mouse_event);
}

void NodesView::mouseReleaseEvent(QMouseEvent* event)
{
	m_dragMove = false;
	QGraphicsView::mouseReleaseEvent(event);
}

void NodesView::wheelEvent(QWheelEvent* wheel_event)
{
	if (QApplication::keyboardModifiers() == _modifiers)
	{
		if (wheel_event->orientation() == Qt::Vertical)
		{
			double angle = wheel_event->angleDelta().y();
			if (angle > 0)
				zoomIn();
			else
				zoomOut();
			return;
		}
	}
	QGraphicsView::wheelEvent(wheel_event);
}

void NodesView::paintEvent(QPaintEvent* event)
{
	QGraphicsView::paintEvent(event);
}

void NodesView::drawForeground(QPainter* painter, const QRectF& rect)
{
	QGraphicsView::drawForeground(painter, rect);
}