#include "zcurvemapeditor.h"
#include <zenoui/nodesys/nodegrid.h>


ZCurveMapView::ZCurveMapView(QWidget* parent)
	: QGraphicsView(parent)
	, _modifiers(Qt::ControlModifier)
	, m_factor(1.)
	, m_dragMove(false)
{
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setDragMode(QGraphicsView::NoDrag);
	setTransformationAnchor(QGraphicsView::NoAnchor);
	setFrameShape(QFrame::NoFrame);
	setMouseTracking(true);
	setContextMenuPolicy(Qt::DefaultContextMenu);
	setBackgroundBrush(QColor(29, 31, 31));
}

ZCurveMapView::~ZCurveMapView()
{

}

void ZCurveMapView::init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	QGraphicsScene* pScene = new QGraphicsScene;
	setScene(pScene);
	setSceneRect(QRectF(QPointF(range.xFrom, range.yFrom), QPointF(range.xTo, range.yTo)));

	//this->setTransform()
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
	QTransform trans = transform();
	QPointF pos = event->pos();
	QPointF scenePos = mapToScene(pos.toPoint());

	QGraphicsView::mousePressEvent(event);
}

void ZCurveMapView::resizeEvent(QResizeEvent* event)
{
	QGraphicsView::resizeEvent(event);
	event->size();
	setSceneRect(QRectF(QPointF(0, 0), QPointF(1, 1)));
}

void ZCurveMapView::drawBackground(QPainter* painter, const QRectF& rect)
{
	QGraphicsView::drawBackground(painter, rect);
	drawGrid(painter, rect);
}

void ZCurveMapView::gentle_zoom(qreal factor)
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
	int innerGrid = 1;   //will be associated with scale factor.

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
	pen.setColor(QColor(32, 32, 33));
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