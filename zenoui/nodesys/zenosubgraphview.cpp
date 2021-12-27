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

	m_ctrlz = new QAction("Undo", this);
    m_ctrlz->setShortcut(QKeySequence::Delete);
    connect(m_ctrlz, SIGNAL(triggered()), this, SLOT(undo()));
    m_ctrly = new QAction("Redo", this);
    m_ctrly->setShortcut(QKeySequence::Redo);
    connect(m_ctrly, SIGNAL(triggered()), this, SLOT(redo()));

	addAction(m_ctrlz);
    addAction(m_ctrly);
}

void ZenoSubGraphView::redo()
{
    m_model->redo();
}

void ZenoSubGraphView::undo()
{
    m_model->undo();
}

void ZenoSubGraphView::setModel(SubGraphModel* pModel)
{
    m_scene = new ZenoSubGraphScene(this);
    m_model = pModel;
    m_scene->initModel(pModel);
    setScene(m_scene);
}

void ZenoSubGraphView::gentle_zoom(qreal factor)
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
	m_factor = std::min(m_factor + _factorStep(m_factor), 32.0);
	qreal current_factor = transform().m11();
	qreal factor_complicate = m_factor / current_factor;
	gentle_zoom(factor_complicate);
}

void ZenoSubGraphView::zoomOut()
{
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

void ZenoSubGraphView::mousePressEvent(QMouseEvent* event)
{
	if (event->type() == QEvent::MouseButtonPress &&
		QApplication::keyboardModifiers() == _modifiers)
	{
		m_dragMove = true;
		m_startPos = event->pos();
	}
    if (event->button() == Qt::LeftButton) {
        setDragMode(QGraphicsView::RubberBandDrag);
    }
	QGraphicsView::mousePressEvent(event);
}

void ZenoSubGraphView::mouseMoveEvent(QMouseEvent* mouse_event)
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

void ZenoSubGraphView::mouseReleaseEvent(QMouseEvent* event)
{
	m_dragMove = false;
    setDragMode(QGraphicsView::NoDrag);
	QGraphicsView::mouseReleaseEvent(event);
}

void ZenoSubGraphView::wheelEvent(QWheelEvent* wheel_event)
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