#include "curvemapview.h"
#include "curvescalaritem.h"
#include "curvegrid.h"
#include "curvenodeitem.h"
#include "curveutil.h"
#include "../model/curvemodel.h"
#include "../util/log.h"


CurveMapView::CurveMapView(QWidget* parent)
	: QGraphicsView(parent)
	, _modifiers(Qt::ControlModifier)
	, m_factor(1.)
	, m_dragMove(false)
	, m_pHScalar(nullptr)
	, m_pVScalar(nullptr)
	, m_grid(nullptr)
	, m_bSmoothCurve(true)
	, m_range({0,1,0,1})
{
	setRenderHint(QPainter::Antialiasing);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);//it's easy but not efficient
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setDragMode(QGraphicsView::NoDrag);
	setTransformationAnchor(QGraphicsView::NoAnchor);
	setFrameShape(QFrame::NoFrame);
	setMouseTracking(true);
	setContextMenuPolicy(Qt::DefaultContextMenu);
	setBackgroundBrush(QColor(31, 31, 31));
}

CurveMapView::~CurveMapView()
{
}

void CurveMapView::init(bool timeFrame)
{
	QGraphicsScene* pScene = new QGraphicsScene;
	setScene(pScene);

	m_gridMargins.setLeft(64);
	m_gridMargins.setRight(64);
	m_gridMargins.setTop(64);
	m_gridMargins.setBottom(64);

	m_pHScalar = new CurveScalarItem(true, timeFrame, this);
	m_pVScalar = new CurveScalarItem(false, false, this);
	connect(horizontalScrollBar(), SIGNAL(valueChanged(int)), m_pHScalar, SLOT(update()));
	connect(verticalScrollBar(), SIGNAL(valueChanged(int)), m_pVScalar, SLOT(update()));
    connect(pScene, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
    connect(m_pHScalar, SIGNAL(frameChanged(qreal)), this, SIGNAL(frameChanged(qreal)));

	QRectF rc = scene()->sceneRect();
	rc = sceneRect();

	m_fixedSceneRect = curve_util::initGridSize(QSize(512, 512), m_gridMargins);

	m_grid = new CurveGrid(this, m_fixedSceneRect);
	m_grid->setColor(QColor(32, 32, 32), QColor(22, 22, 24));
	m_grid->setZValue(-100);

	pScene->addItem(m_pHScalar);
	pScene->addItem(m_pVScalar);
	pScene->addItem(m_grid);
}

void CurveMapView::addCurve(CurveModel* model)
{
    ZASSERT_EXIT(m_grid && model);

	//todo: union range.
    m_range = model->range();
    m_grid->resetTransform(m_fixedSceneRect.marginsRemoved(m_gridMargins), m_range);
    m_grid->addCurve(model);
    m_pHScalar->update();
    m_pVScalar->update();
}

CURVE_RANGE CurveMapView::range() const
{
    return m_range;
}

QPointF CurveMapView::mapLogicToScene(const QPointF& logicPos)
{
	const QRectF& bbox = gridBoundingRect();
	qreal x = logicPos.x(), y = logicPos.y();
	qreal sceneX = bbox.width() * (x - m_range.xFrom) / (m_range.xTo - m_range.xFrom) + bbox.left();
	qreal sceneY = bbox.height() * (m_range.yTo - y) / (m_range.yTo - m_range.yFrom) + bbox.top();
	return QPointF(sceneX, sceneY);
}

QPointF CurveMapView::mapSceneToLogic(const QPointF& scenePos)
{
	const QRectF& bbox = gridBoundingRect();
	qreal x = (m_range.xTo - m_range.xFrom) * (scenePos.x() - bbox.left()) / bbox.width() + m_range.xFrom;
	qreal y = m_range.yTo - (m_range.yTo - m_range.yFrom) * (scenePos.y() - bbox.top()) / bbox.height();
	return QPointF(x, y);
}

void CurveMapView::gentle_zoom(qreal factor)
{
	//scale.
	//QTransform matrix = transform();
	//matrix.scale(factor, factor);
	//if (matrix.m11() < 1.0)
	//	return;
	//setTransform(matrix);

	scale(factor, factor);

	centerOn(target_scene_pos);
	QPointF delta_viewport_pos = target_viewport_pos -
		QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
	QPointF viewport_center = mapFromScene(target_scene_pos) - delta_viewport_pos;
	centerOn(mapToScene(viewport_center.toPoint()));

	m_factor = transform().m11();

	if (m_pHScalar && m_pVScalar)
	{
        m_pHScalar->update();
        m_pVScalar->update();
	}
}

void CurveMapView::resizeEvent(QResizeEvent* event)
{
	QGraphicsView::resizeEvent(event);
	QRectF rc = sceneRect();
	setSceneRect(m_fixedSceneRect);
	
	fitInView(m_fixedSceneRect, Qt::IgnoreAspectRatio);
	if (m_pHScalar && m_pVScalar)
	{
        m_pHScalar->update();
        m_pVScalar->update();
	}
}

CurveGrid* CurveMapView::gridItem() const
{
    return m_grid;
}

QList<CurveNodeItem*> CurveMapView::getSelectedNodes()
{
	auto selItems = scene()->selectedItems();
	QList<CurveNodeItem*> lstNodes;
    for (auto item : selItems)
	{
        if (CurveNodeItem* pNode = qgraphicsitem_cast<CurveNodeItem*>(item))
		{
            lstNodes.append(pNode);
        }
		else if (CurveHandlerItem* pHandle = qgraphicsitem_cast<CurveHandlerItem*>(item))
        {
            lstNodes.append(pHandle->nodeItem());
        }
    }
    return lstNodes;
}

void CurveMapView::onSelectionChanged()
{
    emit nodeItemsSelectionChanged(getSelectedNodes());
}

QRectF CurveMapView::gridBoundingRect() const
{
	QRectF rc = rect();
	rc = rc.marginsRemoved(m_gridMargins);
	QRectF rc2 = m_grid->boundingRect();
	return rc2;
}

void CurveMapView::wheelEvent(QWheelEvent* event)
{
	qreal zoomFactor = 1;
	if (event->angleDelta().y() > 0)
		zoomFactor = 1.25;
	else if (event->angleDelta().y() < 0)
		zoomFactor = 1 / 1.25;
	gentle_zoom(zoomFactor);
}

void CurveMapView::mousePressEvent(QMouseEvent* event)
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
		QRectF rc = scene()->sceneRect();
		rc = sceneRect();
		QPointF logicPos = mapSceneToLogic(scenePos);
		QPointF scenePos2 = mapLogicToScene(logicPos);
		if (scenePos != scenePos2)
		{
			int j;
			j = 0;
		}
		setDragMode(QGraphicsView::RubberBandDrag);
	}
	QGraphicsView::mousePressEvent(event);
}

void CurveMapView::mouseMoveEvent(QMouseEvent* event)
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

		if (m_pHScalar && m_pVScalar)
		{
			m_pHScalar->update();
			m_pVScalar->update();
		}
	}
	QGraphicsView::mouseMoveEvent(event);
}

void CurveMapView::mouseReleaseEvent(QMouseEvent* event)
{
	QGraphicsView::mouseReleaseEvent(event);
	if (event->button() == Qt::MidButton)
	{
		m_dragMove = false;
		setDragMode(QGraphicsView::NoDrag);
	}
}

bool CurveMapView::isSmoothCurve() const
{
    return m_bSmoothCurve;
}

void CurveMapView::drawBackground(QPainter* painter, const QRectF& rect)
{
	QGraphicsView::drawBackground(painter, rect);
}



void CurveMapView::set_modifiers(Qt::KeyboardModifiers modifiers)
{
	_modifiers = modifiers;
}

void CurveMapView::resetTransform()
{
	QGraphicsView::resetTransform();
	m_factor = 1.0;
}
