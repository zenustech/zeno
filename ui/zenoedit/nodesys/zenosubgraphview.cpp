#include "zenosubgraphscene.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenosubgraphview.h"
#include "zenosearchbar.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zenonode.h"
#include "zenonewmenu.h"
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/comctrl/ziconbutton.h>
#include <zenoui/comctrl/gv/zgraphicstextitem.h>
#include "common_def.h"
#include <zeno/utils/cppdemangle.h>
#include "viewport/zenovis.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "util/log.h"
#include <zenomodel/include/uihelper.h>
#include "settings/zenosettingsmanager.h"
#include "groupnode.h"
#include "viewport/cameracontrol.h"


bool sceneMenuEvent(
    ZenoSubGraphScene* pScene,
    const QPointF& pos,
    const QPointF& scenePos,
    const QList<QGraphicsItem*>& seledItems,
    const QList<QGraphicsItem*>& items,
    const QModelIndex& subgIdx);


_ZenoSubGraphView::_ZenoSubGraphView(QWidget *parent)
    : QGraphicsView(parent)
    , m_scene(nullptr)
    , _modifiers(Qt::ControlModifier)
    , m_factor(1.)
    , m_dragMove(false)
    , m_menu(nullptr)
    , m_pSearcher(nullptr)
{
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);//it's easy but not efficient
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setFrameShape(QFrame::NoFrame);
    viewport()->installEventFilter(this);
    setMouseTracking(true);
    setContextMenuPolicy(Qt::DefaultContextMenu);

    //QAction* ctrlz = new QAction("Undo", this);
 //   ctrlz->setShortcut(QKeySequence::Undo);
    //ctrlz->setShortcutContext(Qt::WidgetShortcut);
 //   connect(ctrlz, SIGNAL(triggered()), this, SLOT(undo()));
 //   addAction(ctrlz);

 //   QAction* ctrly = new QAction("Redo", this);
 //   ctrly->setShortcut(QKeySequence::Redo);
    //ctrly->setShortcutContext(Qt::WidgetShortcut);
 //   connect(ctrly, SIGNAL(triggered()), this, SLOT(redo()));
 //   addAction(ctrly);

    QAction *ctrlc = new QAction("Copy", this);
    ctrlc->setShortcut(QKeySequence::Copy);
    ctrlc->setShortcutContext(Qt::WidgetShortcut);
    connect(ctrlc, SIGNAL(triggered()), this, SLOT(copy()));
    addAction(ctrlc);

    QAction *ctrlv = new QAction("Paste", this);
    ctrlv->setShortcut(QKeySequence::Paste);
    ctrlv->setShortcutContext(Qt::WidgetShortcut);
    connect(ctrlv, SIGNAL(triggered()), this, SLOT(paste()));
    addAction(ctrlv);

    QAction *ctrlf = new QAction("Find", this);
    ctrlf->setShortcut(QKeySequence::Find);
    ctrlf->setShortcutContext(Qt::WidgetShortcut);
    connect(ctrlf, SIGNAL(triggered()), this, SLOT(find()));
    addAction(ctrlf);

    QAction* escape = new QAction("Esc", this);
    escape->setShortcut(QKeySequence("Escape"));
    escape->setShortcutContext(Qt::WidgetShortcut);
    connect(escape, SIGNAL(triggered()), this, SLOT(esc()));
    addAction(escape);

    ZenoSettingsManager &settings = ZenoSettingsManager::GetInstance();
    QAction* cameraFocus = new QAction("CameraFocus", this);
    cameraFocus->setShortcut(settings.getShortCut(ShortCut_Focus));
    cameraFocus->setShortcutContext(Qt::WidgetShortcut);
    connect(cameraFocus, SIGNAL(triggered()), this, SLOT(cameraFocus()));
    addAction(cameraFocus);

    QAction* mActZenoNewNode = new QAction();
    mActZenoNewNode->setShortcut(settings.getShortCut(ShortCut_NewNode));
    connect(mActZenoNewNode, &QAction::triggered, [=]() {
        QPoint pos = this->mapFromGlobal(QCursor::pos());
        QContextMenuEvent *e = new QContextMenuEvent(QContextMenuEvent::Reason::Mouse, pos, QCursor::pos());
        contextMenuEvent(e);
    });
    addAction(mActZenoNewNode);

    connect(&ZenoSettingsManager::GetInstance(), &ZenoSettingsManager::valueChanged, this, [=](QString name) {
        if (name == zsShowGrid && isVisible()) {
            showGrid(ZenoSettingsManager::GetInstance().getValue(name).toBool());
        } else if (name == ShortCut_NewNode) {
            mActZenoNewNode->setShortcut(ZenoSettingsManager::GetInstance().getShortCut(ShortCut_NewNode));
        } else if (name == ShortCut_Focus) {
            cameraFocus->setShortcut(ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Focus));
        }
    });

    QRectF rcView(-SCENE_INIT_WIDTH / 2, -SCENE_INIT_HEIGHT / 2, SCENE_INIT_WIDTH, SCENE_INIT_HEIGHT);
    setSceneRect(rcView);

    gentle_zoom(1.0);
}

void _ZenoSubGraphView::showGrid(bool bShow)
{
    scene()->invalidate(rect());
}

void _ZenoSubGraphView::redo()
{
    m_scene->redo();
}

void _ZenoSubGraphView::undo()
{
    m_scene->undo();
}

void _ZenoSubGraphView::copy()
{
    m_scene->copy();
}

void _ZenoSubGraphView::paste()
{
    QPointF pos = mapToScene(m_mousePos);
    m_scene->paste(pos);
}

void _ZenoSubGraphView::find()
{
    if (m_pSearcher == nullptr)
    {
        m_pSearcher = new ZenoSearchBar(m_scene->subGraphIndex(), this);
        connect(m_pSearcher, SIGNAL(searchReached(SEARCH_RECORD)), this, SLOT(onSearchResult(SEARCH_RECORD)));

        QAction* findNext = new QAction("Find Next", this);
        findNext->setShortcut(QKeySequence::FindNext);
        findNext->setShortcutContext(Qt::WidgetShortcut);
        connect(findNext, SIGNAL(triggered()), m_pSearcher, SLOT(onSearchForward()));
        addAction(findNext);

        QAction* findPrevious = new QAction("Find Previous", this);
        findPrevious->setShortcut(QKeySequence::FindPrevious);
        findPrevious->setShortcutContext(Qt::WidgetShortcut);
        connect(findPrevious, SIGNAL(triggered()), m_pSearcher, SLOT(onSearchBackward()));
        addAction(findPrevious);
    }
    m_pSearcher->activate();
}

void _ZenoSubGraphView::esc()
{
    if (m_pSearcher)
        m_pSearcher->hide();
}

void _ZenoSubGraphView::cameraFocus()
{
    QList<QGraphicsItem*> selItems = m_scene->selectedItems();
    if (selItems.size() != 1) {
        return;
    }
    QGraphicsItem* item = selItems[0];
    if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item)) {
        QString nodeId = pNode->nodeId();
        zeno::vec3f center;
        float radius;

        ZenoMainWindow *pWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pWin);
        QVector<DisplayWidget*> views = pWin->viewports();
        for (auto pDisplay : views)
        {
            ZASSERT_EXIT(pDisplay);
            auto pZenovis = pDisplay->getZenoVis();
            ZASSERT_EXIT(pZenovis);

            bool found = pZenovis->getSession()->focus_on_node(nodeId.toStdString(), center, radius);
            if (found) {
                pZenovis->m_camera_control->focus(QVector3D(center[0], center[1], center[2]), radius * 3.0f);
                zenoApp->getMainWindow()->updateViewport();
            }
        }
    }
}

void _ZenoSubGraphView::onSearchResult(SEARCH_RECORD rec)
{
    focusOn(rec.id, rec.pos, false);
}

void _ZenoSubGraphView::focusOnWithNoSelect(const QString& nodeId)
{
    QGraphicsItem* pItem = m_scene->getNode(nodeId);
    if (pItem)
    {
        QRectF rcBounding = pItem->sceneBoundingRect();
        rcBounding.adjust(-rcBounding.width(), -rcBounding.height(), rcBounding.width() / 10., rcBounding.height() / 10.);
        fitInView(rcBounding, Qt::KeepAspectRatio);
    }
}

void _ZenoSubGraphView::focusOn(const QString& nodeId, const QPointF& pos, bool isError)
{
    if (isError)
        m_scene->markError(nodeId);
    else
        m_scene->select(nodeId);

    auto items = m_scene->selectedItems();
    for (auto item : items)
    {
        if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
        {
            QRectF rcBounding = pNode->sceneBoundingRect();
            //rcBounding.adjust(-rcBounding.width(), -rcBounding.height(), rcBounding.width(), rcBounding.height());
            //fitInView(rcBounding, Qt::KeepAspectRatio);
            target_scene_pos = rcBounding.center();
            //editor_factor = transform().m11();
            //emit zoomed(editor_factor);
            gentle_zoom(1.0);
            centerOn(target_scene_pos);
        }
    }
}

void _ZenoSubGraphView::initScene(ZenoSubGraphScene* pScene)
{
    if (!pScene)
        return;

    m_scene = pScene;
    setScene(m_scene);
    QRectF rect = m_scene->nodesBoundingRect();
    fitInView(rect, Qt::KeepAspectRatio);
    target_scene_pos = rect.center();
    editor_factor = transform().m11();
    emit zoomed(editor_factor);
}

void _ZenoSubGraphView::setPath(const QString& path)
{
    m_path = path;
    update();
}

void _ZenoSubGraphView::scaleBy(qreal scaleFactor)
{
    qreal curScaleFactor = transform().m11();
    QVector<qreal> factors = UiHelper::scaleFactors();
    qreal minScale = factors.first();
    qreal maxScale = factors.last();

    if (((curScaleFactor == minScale) && (scaleFactor < 1.0)) ||
        ((curScaleFactor == maxScale) && (scaleFactor > 1.0))) return;

    qreal sc = scaleFactor;
    if ((curScaleFactor * sc < minScale) && (sc < 1.0))
    {
        sc = minScale / curScaleFactor;
    }
    else if ((curScaleFactor * sc > maxScale) && (sc > 1.0))
    {
        sc = maxScale / curScaleFactor;
    }
    scale(sc, sc);
    //centerOn(target_scene_pos);
    editor_factor = transform().m11();
    m_scene->onZoomed(editor_factor);
}

void _ZenoSubGraphView::setScale(qreal scale)
{
    qreal scaleFactor = scale / transform().m11();
    scaleBy(scaleFactor);
}

void _ZenoSubGraphView::gentle_zoom(qreal factor)
{
    //scale(factor, factor);
    QTransform matrix = this->transform();
    matrix.setMatrix(factor,        matrix.m12(),    matrix.m13(),
                     matrix.m21(),    factor,            matrix.m23(),
                     matrix.m31(),    matrix.m32(),    matrix.m33());
    setTransform(matrix);

    centerOn(target_scene_pos);
    QPointF delta_viewport_pos = target_viewport_pos - 
        QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
    QPointF viewport_center = mapFromScene(target_scene_pos) - delta_viewport_pos;
    centerOn(mapToScene(viewport_center.toPoint()));

    qreal factor_i_want = transform().m11();
    editor_factor = factor_i_want;    //temp: test factor
    emit zoomed(factor_i_want);
    emit viewChanged(m_factor);
}

qreal _ZenoSubGraphView::scaleFactor() const
{
    qreal factor_i_want = transform().m11();
    return factor_i_want;
}

void _ZenoSubGraphView::set_modifiers(Qt::KeyboardModifiers modifiers)
{
    _modifiers = modifiers;
}

void _ZenoSubGraphView::resetTransform()
{
    QGraphicsView::resetTransform();
    m_factor = 1.0;
}

void _ZenoSubGraphView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
}

void _ZenoSubGraphView::showEvent(QShowEvent *event) 
{
    qreal factor_i_want = transform().m11();
    if (factor_i_want != editor_factor) 
    {
        editor_factor = factor_i_want;
        emit zoomed(factor_i_want);
    }
    _base::showEvent(event);
}

void _ZenoSubGraphView::mousePressEvent(QMouseEvent* event)
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

void _ZenoSubGraphView::mouseMoveEvent(QMouseEvent* event)
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

void _ZenoSubGraphView::mouseReleaseEvent(QMouseEvent* event)
{
    QGraphicsView::mouseReleaseEvent(event);
    if (event->button() == Qt::MidButton)
    {
        m_dragMove = false;
        setDragMode(QGraphicsView::NoDrag);
    }
}

void _ZenoSubGraphView::mouseDoubleClickEvent(QMouseEvent* event)
{
    QGraphicsView::mouseDoubleClickEvent(event);
}

void _ZenoSubGraphView::wheelEvent(QWheelEvent* event)
{
    //mock QGraphicsView::wheelEvent:
    event->ignore();
    QGraphicsSceneWheelEvent wheelEvent(QEvent::GraphicsSceneWheel);
    wheelEvent.setWidget(viewport());

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    wheelEvent.setScenePos(mapToScene(event->position().toPoint()));
    wheelEvent.setScreenPos(event->globalPosition().toPoint());
#else
    wheelEvent.setScenePos(mapToScene(event->posF().toPoint()));
    wheelEvent.setScreenPos(event->globalPosF().toPoint());
#endif

    wheelEvent.setButtons(event->buttons());
    wheelEvent.setModifiers(event->modifiers());
    const bool horizontal = qAbs(event->angleDelta().x()) > qAbs(event->angleDelta().y());
    wheelEvent.setDelta(horizontal ? event->angleDelta().x() : event->angleDelta().y());
    wheelEvent.setOrientation(horizontal ? Qt::Horizontal : Qt::Vertical);
    wheelEvent.setAccepted(false);
    QCoreApplication::sendEvent(scene(), &wheelEvent);
    event->setAccepted(wheelEvent.isAccepted());
    if (!event->isAccepted())
    {
        //executing zoom
        QVector<qreal> factors = UiHelper::scaleFactors();
        qreal zoomFactor = transform().m11();
        qreal spread = zoomFactor / 5;
        if (event->angleDelta().y() > 0)
            zoomFactor += spread;
        else if (event->angleDelta().y() < 0)
            zoomFactor -= spread;

       if (zoomFactor < factors.first()) {
            zoomFactor = factors.first();
        } else if (zoomFactor > factors.last()) {
            zoomFactor = factors.last();
        }
        gentle_zoom(zoomFactor);
    }
}

bool _ZenoSubGraphView::eventFilter(QObject* watched, QEvent* event)
{
    return QGraphicsView::eventFilter(watched, event);
}

void _ZenoSubGraphView::focusOutEvent(QFocusEvent* event)
{
    if (m_dragMove)
    {
        m_dragMove = false;
        setDragMode(QGraphicsView::NoDrag);
    }
    QGraphicsView::focusOutEvent(event);
}

void _ZenoSubGraphView::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);

    if (m_pSearcher)
    {
        QSize sz = m_pSearcher->size();
        int w = width();
        int h = height();
        m_pSearcher->setGeometry(w - sz.width(), 0, sz.width(), sz.height());
    }
}

void _ZenoSubGraphView::contextMenuEvent(QContextMenuEvent* event)
{
    QPoint pos = event->pos();
    QPointF scenePos = mapToScene(pos);

    QList<QGraphicsItem*> seledItems = m_scene->selectedItems();
    QList<QGraphicsItem*> tempList = this->items(pos);

    bool ret = sceneMenuEvent(m_scene, pos, scenePos, seledItems, tempList, m_scene->subGraphIndex());
    if (!ret) {
        QGraphicsView::contextMenuEvent(event);
    }

}

void _ZenoSubGraphView::drawGrid(QPainter* painter, const QRectF& rect)
{
    //background color
    painter->fillRect(rect, QColor("#1d1d1d"));
    bool showGrid = ZenoSettingsManager::GetInstance().getValue(zsShowGrid).toBool();
    if (showGrid)
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

        if (scale >= 0.1)
        {
            QPen pen;
            pen.setColor(QColor("#1F2933"));
            pen.setWidthF(pen.widthF() / scale);
            painter->setPen(pen);
            painter->drawLines(innerLines.data(), innerLines.size());
        }
    }
}

void _ZenoSubGraphView::drawBackground(QPainter* painter, const QRectF& rect)
{
    drawGrid(painter, rect);
}


//////////////////////////////////////////////////////////////////////////////////
LayerPathWidget::LayerPathWidget(QWidget* parent)
    : QWidget(parent)
{
    QHBoxLayout* pLayout = new QHBoxLayout;
    pLayout->setSpacing(10);
    pLayout->setContentsMargins(25, 5, 25, 5);
    setLayout(pLayout);

    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(30, 30, 30));
    setPalette(pal);
}

void LayerPathWidget::setPath(const QString& path)
{
    if (m_path == path)
        return;

    m_path = path;
    QHBoxLayout* pLayout = qobject_cast<QHBoxLayout*>(this->layout());
    while (pLayout->count() > 0)
    {
        QLayoutItem* pItem = pLayout->itemAt(pLayout->count() - 1);
        pLayout->removeItem(pItem);
    }

    QStringList L = m_path.split("/", QtSkipEmptyParts);
    for (int i = 0; i < L.length(); i++)
    {
        const QString& item = L[i];
        ZASSERT_EXIT(!item.isEmpty());
        QColor clrHovered, clrSelected;
        clrHovered = QColor(67, 67, 67);
        clrSelected = QColor(33, 33, 33);

        ZTextLabel* pLabel = new ZTextLabel;
        pLabel->setText(item);
        QFont font = QApplication::font();
        font.setPointSize(11);
        pLabel->setFont(font);
        pLabel->setTextColor(QColor(129, 125, 123));
        connect(pLabel, SIGNAL(clicked()), this, SLOT(onPathItemClicked()));
        pLayout->addWidget(pLabel);

        if (L.indexOf(item) != L.length() - 1)
        {
            pLabel = new ZTextLabel;
            pLabel->setText(">");
            QFont font = QApplication::font();
            font.setPointSize(11);
            font.setBold(true);
            pLabel->setFont(font);
            pLabel->setTextColor(QColor(129, 125, 123));
            pLayout->addWidget(pLabel);
        }
    }
    pLayout->addStretch();
    update();
}

QString LayerPathWidget::path() const
{
    return m_path;
}

void LayerPathWidget::onPathItemClicked()
{
    ZTextLabel* pClicked = qobject_cast<ZTextLabel*>(sender());
    QString path;
    QHBoxLayout* pLayout = qobject_cast<QHBoxLayout*>(this->layout());

    bool bStartDeleted = false;
    for (int i = 0; i < pLayout->count(); i++)
    {
        QLayoutItem* pItem = pLayout->itemAt(i);
        QWidget* w = pItem->widget();
        if (ZTextLabel* pPathItem = qobject_cast<ZTextLabel*>(w))
        {
            if (pPathItem->text() != '>')
            {
                path += "/" + pPathItem->text();
                if (pPathItem == pClicked)
                    break;
            }
        }
    }
    emit pathUpdated(path);
}


ZenoSubGraphView::ZenoSubGraphView(QWidget* parent)
    : QWidget(parent)
    , m_prop(nullptr)
    , m_floatPanelShow(false)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setSpacing(0);
    pLayout->setContentsMargins(0, 0, 0, 0);

    m_pathWidget = new LayerPathWidget;
    m_pathWidget->hide();
    pLayout->addWidget(m_pathWidget);

    m_view = new _ZenoSubGraphView;
    pLayout->addWidget(m_view);

    setLayout(pLayout);

    connect(m_pathWidget, SIGNAL(pathUpdated(QString)), this, SIGNAL(pathUpdated(QString)));
    connect(m_view, SIGNAL(zoomed(qreal)), this, SIGNAL(zoomed(qreal)));
}

void ZenoSubGraphView::initScene(ZenoSubGraphScene* pScene)
{
    m_view->initScene(pScene);
}

ZenoSubGraphScene* ZenoSubGraphView::scene()
{
    return qobject_cast<ZenoSubGraphScene*>(m_view->scene());
}

void ZenoSubGraphView::resetPath(const QString& path, const QString& subGraphName, const QString& objId, bool isError)
{
    if (path.isEmpty())
    {
        m_pathWidget->hide();
    }
    else
    {
        m_pathWidget->show();
        m_pathWidget->setPath(path);
    }
    if (!subGraphName.isEmpty() && !objId.isEmpty())
    {
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        QModelIndex subgIdx = pModel->index(subGraphName);
        QModelIndex objIdx = pModel->index(objId, subgIdx);
        ZASSERT_EXIT(objIdx.isValid());
        QPointF pos = objIdx.data(ROLE_OBJPOS).toPointF();
        m_view->focusOn(objId, pos, isError);
    }
}

void ZenoSubGraphView::setZoom(const qreal& scale)
{
    m_view->setScale(scale);
}

void ZenoSubGraphView::focusOnWithNoSelect(const QString& nodeId)
{
    m_view->focusOnWithNoSelect(nodeId);
}

void ZenoSubGraphView::focusOn(const QString& nodeId)
{
    m_view->focusOn(nodeId, QPointF(), false);
}

void ZenoSubGraphView::showFloatPanel(const QModelIndex &subgIdx, const QModelIndexList &nodes) {
    if (m_floatPanelShow) {
        if (nodes.isEmpty() && m_prop) {
            m_prop->onNodesSelected(subgIdx, nodes, true);
            m_lastSelectedNode = QModelIndex();
        } else if (m_prop == nullptr || nodes[0] != m_lastSelectedNode) {
            if (m_prop == nullptr) {
                m_prop = new DockContent_Parameter(this);
                m_prop->initUI();
                m_prop->resize(this->width() * 0.2, this->height());
                m_prop->setMinimumWidth(300);
                m_prop->setMinimumHeight(400);
            }
            m_prop->show();
            m_prop->onNodesSelected(subgIdx, nodes, true);

            m_lastSelectedNode = nodes[0];
        } else if (!m_prop->isVisible()) {
            m_prop->setVisible(m_floatPanelShow);
        }
        m_prop->move(this->width() - m_prop->width(), 0);
    }
}

void ZenoSubGraphView::selectNodes(const QModelIndexList &indexs) 
{
    ZenoSubGraphScene *scene = qobject_cast<ZenoSubGraphScene *>(m_view->scene());
    if (scene) 
    {
        scene->select(indexs);
    }
}

void ZenoSubGraphView::cameraFocus()
{
    m_view->cameraFocus();
}

void ZenoSubGraphView::keyPressEvent(QKeyEvent *event) {
    int uKey = event->key();
    Qt::KeyboardModifiers modifiers = event->modifiers();
    if (modifiers & Qt::ShiftModifier) {
        uKey += Qt::SHIFT;
    }
    if (modifiers & Qt::ControlModifier) {
        uKey += Qt::CTRL;
    }
    if (modifiers & Qt::AltModifier) {
        uKey += Qt::ALT;
    }
    if (uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_FloatPanel)) {
        ZenoSubGraphScene *scene = qobject_cast<ZenoSubGraphScene *>(m_view->scene());
        if (scene != NULL)
        {
            if (scene->selectNodesIndice().size() == 1 && (!m_prop || m_prop->isHidden())) {
                m_floatPanelShow = true;
                showFloatPanel(scene->subGraphIndex(), scene->selectNodesIndice());
            } else if (/*scene->selectNodesIndice().size() == 0 && */m_prop != NULL && m_prop->isVisible()) {
                m_prop->hide();
                m_floatPanelShow = false;
            }
        }
    }
    QWidget::keyPressEvent(event);
}

void ZenoSubGraphView::resizeEvent(QResizeEvent *event) {
    if (m_prop != nullptr && m_prop->isVisible()) {
        m_prop->resize(m_prop->width(), this->height() * 1.0);
        m_prop->move(this->width() - m_prop->width(), 0);
    }
    QWidget::resizeEvent(event);
}
