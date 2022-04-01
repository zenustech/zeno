#include "curvemapview.h"
#include "curvescalaritem.h"
#include "curvegrid.h"


ZCurveMapView::ZCurveMapView(QWidget* parent)
	: QGraphicsView(parent)
	, _modifiers(Qt::ControlModifier)
	, m_factor(1.)
	, m_dragMove(false)
	, m_pHScalar(nullptr)
	, m_pVScalar(nullptr)
	, m_grid(nullptr)
{
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);//it's easy but not efficient
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setDragMode(QGraphicsView::NoDrag);
	setTransformationAnchor(QGraphicsView::NoAnchor);
	setFrameShape(QFrame::NoFrame);
	setMouseTracking(true);
	setContextMenuPolicy(Qt::DefaultContextMenu);
	setBackgroundBrush(QColor(26, 26, 26));
}

ZCurveMapView::~ZCurveMapView()
{

}

void ZCurveMapView::init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	QGraphicsScene* pScene = new QGraphicsScene;
	setScene(pScene);
	m_range = range;

	m_gridMargins.setLeft(64);
	m_gridMargins.setRight(64);
	m_gridMargins.setTop(64);
	m_gridMargins.setBottom(64);

	m_pHScalar = new CurveScalarItem(true, this);
	m_pVScalar = new CurveScalarItem(false, this);
	m_grid = new CurveGrid(this);
	m_grid->setColor(QColor(58, 58, 58), QColor(32, 32, 32));
	m_grid->setZValue(-100);
	pScene->addItem(m_pHScalar);
	pScene->addItem(m_pVScalar);
	pScene->addItem(m_grid);

	connect(horizontalScrollBar(), SIGNAL(valueChanged(int)), m_pHScalar, SLOT(update()));
	connect(verticalScrollBar(), SIGNAL(valueChanged(int)), m_pVScalar, SLOT(update()));
}

void ZCurveMapView::resizeEvent(QResizeEvent* event)
{
	QGraphicsView::resizeEvent(event);
	const QSize sz = event->size();
	setSceneRect(QRectF(0, 0, sz.width(), sz.height()));

	qreal W = sz.width(), H = sz.height();
	qreal W_ = m_range.xTo - m_range.xFrom;
	qreal H_ = m_range.yTo - m_range.yFrom;

	static int margin = 64;
	W = W - 2 * margin;
	H = H - 2 * margin;

	m_pHScalar->setX(m_gridMargins.left());
	m_pVScalar->setY(margin);

	m_pHScalar->update();
	m_pVScalar->update();
}

QRectF ZCurveMapView::gridBoundingRect() const
{
	QRectF rc = rect();
	rc = rc.marginsRemoved(m_gridMargins);
	return rc;
}

void ZCurveMapView::wheelEvent(QWheelEvent* event)
{
	qreal zoomFactor = 1;
	if (event->angleDelta().y() > 0)
		zoomFactor = 1.25;
	else if (event->angleDelta().y() < 0)
		zoomFactor = 1 / 1.25;
	gentle_zoom(zoomFactor);
}

void ZCurveMapView::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::MidButton)
	{
		_last_mouse_pos = event->pos();
		setDragMode(QGraphicsView::NoDrag);
		setDragMode(QGraphicsView::ScrollHandDrag);
		m_dragMove = true;
		QRectF rc = this->sceneRect();
		return;
	}
	if (event->button() == Qt::LeftButton)
	{
		QTransform trans = transform();
		QPointF pos = event->pos();
		QPointF scenePos = mapToScene(pos.toPoint());

		setDragMode(QGraphicsView::RubberBandDrag);
	}
	QGraphicsView::mousePressEvent(event);
}

void ZCurveMapView::mouseMoveEvent(QMouseEvent* event)
{
	m_mousePos = event->pos();
	QPointF delta = target_viewport_pos - m_mousePos;
	if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5)
	{
		target_viewport_pos = m_mousePos;
		target_scene_pos = mapToScene(m_mousePos);
	}
	if (m_dragMove)
	{
		QPointF last_pos = mapToScene(_last_mouse_pos);
		QPointF current_pos = mapToScene(event->pos());
		QPointF delta = last_pos - current_pos;
		translate(-delta.x(), -delta.y());
		_last_mouse_pos = event->pos();

		m_pHScalar->update();
		m_pVScalar->update();
	}
	QGraphicsView::mouseMoveEvent(event);
}

void ZCurveMapView::mouseReleaseEvent(QMouseEvent* event)
{
	QGraphicsView::mouseReleaseEvent(event);
	if (event->button() == Qt::MidButton)
	{
		m_dragMove = false;
		setDragMode(QGraphicsView::NoDrag);
	}
}

int ZCurveMapView::frames(bool bHorizontal) const
{
	//hard to cihou grids...
	if (bHorizontal)
	{
		int W = width();
		int wtf = W * m_factor * 0.015;
		return wtf;
	}
	else
	{
		int H = height();
		int wtf = H * m_factor * 0.015;
		return wtf;
	}
}

void ZCurveMapView::drawBackground(QPainter* painter, const QRectF& rect)
{
	QGraphicsView::drawBackground(painter, rect);
}

void ZCurveMapView::gentle_zoom(qreal factor)
{
	//scale.
	QTransform matrix = transform();
	matrix.scale(factor, factor);
	if (matrix.m11() < 1.0)
		return;
	setTransform(matrix);

	centerOn(target_scene_pos);
	QPointF delta_viewport_pos = target_viewport_pos -
		QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
	QPointF viewport_center = mapFromScene(target_scene_pos) - delta_viewport_pos;
	centerOn(mapToScene(viewport_center.toPoint()));

	m_factor = transform().m11();

	m_pHScalar->update();
	m_pVScalar->update();
	m_grid->update();
}

void ZCurveMapView::set_modifiers(Qt::KeyboardModifiers modifiers)
{
	_modifiers = modifiers;
}

void ZCurveMapView::resetTransform()
{
	QGraphicsView::resetTransform();
	m_factor = 1.0;
}
