#include "zenosubgraphscene.h"
#include "../model/subgraphmodel.h"
#include "zenosubgraphview.h"
#include "zenographswidget.h"


ZenoSubGraphView::ZenoSubGraphView(QWidget *parent)
	: QGraphicsView(parent)
	, m_scene(nullptr)
	, _modifiers(Qt::ControlModifier)
	, m_factor(1.)
	, m_dragMove(false)
	, m_menu(nullptr)
{
    setBackgroundBrush(QBrush(QColor(38, 37, 42), Qt::SolidPattern));
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);//it's easy but not efficient
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    viewport()->installEventFilter(this);
    setMouseTracking(true);
    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(onCustomContextMenu(const QPoint&)));

	QAction* ctrlz = new QAction("Undo", this);
    ctrlz->setShortcut(QKeySequence::Undo);
    connect(ctrlz, SIGNAL(triggered()), this, SLOT(undo()));
    addAction(ctrlz);

    QAction* ctrly = new QAction("Redo", this);
    ctrly->setShortcut(QKeySequence::Redo);
    connect(ctrly, SIGNAL(triggered()), this, SLOT(redo()));
    addAction(ctrly);

	QAction *ctrlc = new QAction("Copy", this);
    ctrlc->setShortcut(QKeySequence::Copy);
    connect(ctrlc, SIGNAL(triggered()), this, SLOT(copy()));
    addAction(ctrlc);

	QAction *ctrlv = new QAction("Paste", this);
    ctrlv->setShortcut(QKeySequence::Paste);
    connect(ctrlv, SIGNAL(triggered()), this, SLOT(paste()));
    addAction(ctrlv);
}

void ZenoSubGraphView::redo()
{
    m_scene->redo();
}

void ZenoSubGraphView::undo()
{
    m_scene->undo();
}

void ZenoSubGraphView::copy()
{
    m_scene->copy();
}

void ZenoSubGraphView::paste()
{
    QPointF pos = mapToScene(m_mousePos);
    m_scene->paste(pos);
}

void ZenoSubGraphView::setModel(SubGraphModel* pModel)
{
    m_scene = new ZenoSubGraphScene(this);
    m_scene->initModel(pModel);
    setScene(m_scene);
    if (!m_scene->_sceneRect().isNull())
        _updateSceneRect();
}

void ZenoSubGraphView::gentle_zoom(qreal factor)
{
    //legacy code.
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

qreal ZenoSubGraphView::_factorStep(qreal factor)
{
	if (factor < 2)
		return 0.2;
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

void ZenoSubGraphView::zoomIn()
{
    //legacy code.
	m_factor = std::min(m_factor + _factorStep(m_factor), 32.0);
	qreal current_factor = transform().m11();
	qreal factor_complicate = m_factor / current_factor;
	gentle_zoom(factor_complicate);
}

void ZenoSubGraphView::zoomOut()
{
    //legacy code.
	m_factor = std::max(m_factor - _factorStep(m_factor), 0.4);
	qreal current_factor = transform().m11();
	qreal factor_complicate = m_factor / current_factor;
	gentle_zoom(factor_complicate);
}

void ZenoSubGraphView::set_modifiers(Qt::KeyboardModifiers modifiers)
{
	_modifiers = modifiers;
}

void ZenoSubGraphView::resetTransform()
{
	QGraphicsView::resetTransform();
	m_factor = 1.0;
}

void ZenoSubGraphView::_updateSceneRect()
{
    QRectF rect = m_scene->_sceneRect();
    setSceneRect(rect);
    fitInView(rect, Qt::KeepAspectRatio);
}

void ZenoSubGraphView::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::MidButton)
	{
        _last_mouse_pos = event->pos();
        setDragMode(QGraphicsView::NoDrag);
        setDragMode(QGraphicsView::ScrollHandDrag);
        m_dragMove = true;
        return;
	}
    if (event->button() == Qt::LeftButton)
	{
        setDragMode(QGraphicsView::RubberBandDrag);
    }
	QGraphicsView::mousePressEvent(event);
}

void ZenoSubGraphView::mouseMoveEvent(QMouseEvent* event)
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
        _last_mouse_pos = event->pos();
        QRectF _sceneRect = m_scene->_sceneRect();
        _sceneRect.translate(delta);
        m_scene->_setSceneRect(_sceneRect);
        _updateSceneRect();
	}
	QGraphicsView::mouseMoveEvent(event);
}

void ZenoSubGraphView::mouseReleaseEvent(QMouseEvent* event)
{
    QGraphicsView::mouseReleaseEvent(event);
	if (event->button() == Qt::MidButton)
    {
        m_dragMove = false;
        setDragMode(QGraphicsView::NoDrag);
	}
}

void ZenoSubGraphView::wheelEvent(QWheelEvent* event)
{
	qreal zoomFactor = 1;
    if (event->angleDelta().y() > 0)
        zoomFactor = 1.25;
    else if (event->angleDelta().y() < 0)
        zoomFactor = 1 / 1.25;

	_scale(zoomFactor, zoomFactor, event->pos());
    _updateSceneRect();

	QGraphicsView::wheelEvent(event);
}

void ZenoSubGraphView::_scale(qreal sx, qreal sy, QPointF pos)
{
    QRectF rect = m_scene->_sceneRect();
    if ((rect.width() > 10000 and sx < 1) || (rect.width() < 200 and sx > 1))
        return;
    pos = mapToScene(pos.x(), pos.y());
    QPointF center = pos;
    qreal w = rect.width() / sx;
    qreal h = rect.height() / sy;
    QRectF rc(center.x() - (center.x() - rect.left()) / sx,
              center.y() - (center.y() - rect.top()) / sy,
              w, h);
    m_scene->_setSceneRect(rc);
    _updateSceneRect();
}

void ZenoSubGraphView::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
}

void ZenoSubGraphView::contextMenuEvent(QContextMenuEvent* event)
{
    QGraphicsView::contextMenuEvent(event);
}

void ZenoSubGraphView::onCustomContextMenu(const QPoint& pos)
{
    if (m_menu)
    {
        delete m_menu;
        m_menu = nullptr;
    }
    m_menu = new QMenu(this);

    ZenoGraphsWidget* pWidget = qobject_cast<ZenoGraphsWidget*>(parent());
    Q_ASSERT(pWidget);

    QList<QAction*> actions = pWidget->getCategoryActions(mapToScene(pos));
    m_menu->addActions(actions);
    m_menu->exec(QCursor::pos());
}