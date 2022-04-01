#include "zcurvemapeditor.h"
#include <zenoui/nodesys/nodegrid.h>
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

	m_pHScalar = new CurveScalarItem(true, this);
	m_pVScalar = new CurveScalarItem(false, this);
	m_grid = new CurveGrid;
	m_grid->setColor(QColor(58, 58, 58), QColor(32, 32, 32));
	m_grid->setZValue(-100);
	pScene->addItem(m_pHScalar);
	pScene->addItem(m_pVScalar);
	pScene->addItem(m_grid);

	connect(horizontalScrollBar(), SIGNAL(valueChanged(int)), m_pHScalar, SLOT(resetPosition()));
	connect(verticalScrollBar(), SIGNAL(valueChanged(int)), m_pVScalar, SLOT(resetPosition()));
}

CURVE_RANGE ZCurveMapView::range() const
{
	return m_range;
}

void ZCurveMapView::resizeEvent(QResizeEvent* event)
{
	QGraphicsView::resizeEvent(event);
	const QSize sz = event->size();

	qreal W = sz.width(), H = sz.height();
	qreal W_ = m_range.xTo - m_range.xFrom;
	qreal H_ = m_range.yTo - m_range.yFrom;

	static int margin = 64;
	W = W - 2 * margin;
	H = H - 2 * margin;

	QRectF rcBound(margin, margin, W, H);
	setSceneRect(QRectF(0, 0, sz.width(), sz.height()));
	m_grid->reset(rcBound);

	m_pHScalar->onResizeView(this);
	m_pHScalar->resetPosition(this);
	m_pHScalar->setX(margin);
	m_pVScalar->onResizeView(this);
	m_pVScalar->resetPosition(this);
	m_pVScalar->setY(margin);
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

		m_pHScalar->updateScalar(this, m_factor, metrics(m_factor));
		m_pVScalar->updateScalar(this, m_factor, metrics(m_factor));
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

int ZCurveMapView::metrics(int factor) const
{
	if (factor >= 1.0 && factor <= 2.5)
	{
		return 20;
	}
	else if (factor < 4)
	{
		return 40;
	}
	else if (factor < 6)
	{
		return 60;
	}
	else if (factor < 8)
	{
		return 80;
	}
	else
	{
		return 120;
	}
}

void ZCurveMapView::drawBackground(QPainter* painter, const QRectF& rect)
{
	QGraphicsView::drawBackground(painter, rect);
	//drawGrid(painter, rect);
}

void ZCurveMapView::gentle_zoom(qreal factor)
{
	QTransform matrix = transform();
	matrix.scale(factor, factor);

	if (matrix.m11() < 1.0)
		return;

	setTransform(matrix);

	//scale(factor, factor);
	centerOn(target_scene_pos);
	QPointF delta_viewport_pos = target_viewport_pos -
		QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
	QPointF viewport_center = mapFromScene(target_scene_pos) - delta_viewport_pos;
	centerOn(mapToScene(viewport_center.toPoint()));

	m_factor = transform().m11();
	emit zoomed(m_factor);
	emit viewChanged(m_factor);

	m_pHScalar->updateScalar(this, m_factor, metrics(m_factor));
	m_pVScalar->updateScalar(this, m_factor, metrics(m_factor));
	m_grid->setFactor(m_factor, metrics(m_factor));
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

void ZCurveMapView::drawGrid(QPainter* painter, const QRectF& rect)
{
	QTransform tf = transform();
	qreal scale = tf.m11();
	int innerGrid = 50;   //will be associated with scale factor.

	qreal left = int(rect.left()) - (int(rect.left()) % innerGrid);
	qreal top = int(rect.top()) - (int(rect.top()) % innerGrid);

	QVarLengthArray<QLineF, 100> innerLines;

	for (qreal x = left; x < rect.right(); x += innerGrid)
	{
		innerLines.append(QLineF(x, rect.top(), x, rect.bottom()));
	}
	for (qreal y = top; y < rect.bottom(); y += innerGrid)
	{
		innerLines.append(QLineF(rect.left(), y, rect.right(), y));
	}

	painter->fillRect(rect, QColor(30, 29, 33));

	QPen pen;
	pen.setColor(QColor(116, 116, 116));
	pen.setWidthF(pen.widthF() / scale);
	painter->setPen(pen);
	painter->drawLines(innerLines.data(), innerLines.size());
}


ZCurveMapEditor::ZCurveMapEditor(QWidget* parent)
	: QWidget(parent)
	, m_view(nullptr)
{

}

ZCurveMapEditor::~ZCurveMapEditor()
{

}

void ZCurveMapEditor::init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	QVBoxLayout* pLayout = new QVBoxLayout;
	pLayout->setContentsMargins(0, 0, 0, 0);
	m_view = new ZCurveMapView;
	m_view->init(range, pts, handlers);
	pLayout->addWidget(m_view);
	setLayout(pLayout);
}