#include "zenosubgraphscene.h"
#include <zenoui/model/subgraphmodel.h>
#include "../graphsmanagment.h"
#include "zenosubgraphview.h"
#include "zenographswidget.h"
#include "zenosearchbar.h"
#include "zenoapplication.h"
#include "zenonode.h"
#include "zenonewmenu.h"


ZenoSubGraphView::ZenoSubGraphView(QWidget *parent)
	: QGraphicsView(parent)
	, m_scene(nullptr)
	, _modifiers(Qt::ControlModifier)
	, m_factor(1.)
	, m_dragMove(false)
    , m_menu(nullptr)
{
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);//it's easy but not efficient
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setFrameShape(QFrame::NoFrame);
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

    QAction *ctrlf = new QAction("Find", this);
    ctrlf->setShortcut(QKeySequence::Find);
    connect(ctrlf, SIGNAL(triggered()), this, SLOT(find()));
    addAction(ctrlf);

    QRectF rcView(-SCENE_INIT_WIDTH / 2, -SCENE_INIT_HEIGHT / 2, SCENE_INIT_WIDTH, SCENE_INIT_HEIGHT);
    setSceneRect(rcView);
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

void ZenoSubGraphView::find()
{
    ZenoSearchBar *pSearcher = new ZenoSearchBar(m_scene->subGraphIndex());
    pSearcher->show();
    connect(pSearcher, SIGNAL(searchReached(SEARCH_RECORD)), this, SLOT(onSearchResult(SEARCH_RECORD)));
}

void ZenoSubGraphView::onSearchResult(SEARCH_RECORD rec)
{
    focusOn(rec.id, rec.pos);
}

void ZenoSubGraphView::focusOn(const QString& nodeId, const QPointF& pos)
{
	m_scene->select(nodeId);
    auto items = m_scene->selectedItems();
    for (auto item : items)
    {
        if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
        {
            QRectF rcBounding = pNode->sceneBoundingRect();
            rcBounding.adjust(-rcBounding.width(), -rcBounding.height(), rcBounding.width(), rcBounding.height());
            fitInView(rcBounding, Qt::KeepAspectRatio);
        }
    }
}

void ZenoSubGraphView::initScene(ZenoSubGraphScene* pScene)
{
    m_scene = pScene;
    setScene(m_scene);
    QRectF rect = m_scene->nodesBoundingRect();
    fitInView(rect, Qt::KeepAspectRatio);
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
        translate(-delta.x(), -delta.y());
        _last_mouse_pos = event->pos();
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
    gentle_zoom(zoomFactor);
}

void ZenoSubGraphView::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
}

void ZenoSubGraphView::contextMenuEvent(QContextMenuEvent* event)
{
    QGraphicsView::contextMenuEvent(event);
}

void ZenoSubGraphView::drawBackground(QPainter* painter, const QRectF& rect)
{
    QTransform tf = transform();
    qreal scale = tf.m11();
    int innerGrid = SCENE_GRID_SIZE;   //will be associated with scale factor.

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

    painter->fillRect(rect, QColor(0, 0, 0));

    QPen pen;
    pen.setColor(QColor("#232323"));
    pen.setWidthF(pen.widthF() / scale);
    painter->setPen(pen);
    painter->drawLines(innerLines.data(), innerLines.size());
}

void ZenoSubGraphView::onCustomContextMenu(const QPoint& pos)
{
    delete m_menu;
    m_menu = nullptr;

    NODE_CATES cates = zenoApp->graphsManagment()->currentModel()->getCates();
    m_menu = new ZenoNewnodeMenu(m_scene->subGraphIndex(), cates, mapToScene(pos), this);
    m_menu->exec(QCursor::pos());
}