#include "thumbnailscene.h"
#include "zenosubgraphscene.h"
#include "model/GraphModel.h"
#include "zenonodebase.h"
#include "zenonode.h"
#include "zassert.h"


NavigatorItem::NavigatorItem(QRectF rcView, qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent)
    : QGraphicsRectItem(x,y,w,h,parent)
    , m_rcView(rcView)
{
    setFlags(ItemSendsGeometryChanges | ItemSendsScenePositionChanges | ItemIsMovable);

    QPen pen(QColor(255, 0, 0), cPenWidth);
    pen.setJoinStyle(Qt::MiterJoin);
    setPen(pen);
    setBrush(Qt::NoBrush);
    setZValue(10);
}

void NavigatorItem::resize(bool bZoomOut)
{
    if (bZoomOut)
    {
        QRectF rcOld = boundingRect();
        QPointF center = pos() + QPointF(rcOld.width() / 2., rcOld.height() / 2.);
        QRectF rcNew(0, 0, rcOld.width() * 1.2, rcOld.height() * 1.2);
        if (rcNew.width() < m_rcView.width() * 0.9)
        {
            setRect(rcNew);
            setPos(center - QPointF(rcNew.width() / 2., rcNew.height() / 2.));
        }
    }
    else {
        QRectF rcOld = boundingRect();
        QPointF center = pos() + QPointF(rcOld.width() / 2., rcOld.height() / 2.);
        QRectF rcNew(0, 0, rcOld.width() * 0.8, rcOld.height() * 0.8);
        if (rcNew.width() > m_rcView.width() * 0.2)
        {
            setRect(rcNew);
            setPos(center - QPointF(rcNew.width() / 2., rcNew.height() / 2.));
        }
    }
}

QVariant NavigatorItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemPositionChange)
    {
        QPointF pos = value.toPointF();
        QRectF rcBound = boundingRect();
        pos.setX(qMax(pos.x(), cPenWidth));
        pos.setY(qMax(pos.y(), cPenWidth));
        pos.setX(qMin(pos.x(), m_rcView.right() - rcBound.width()));
        pos.setY(qMin(pos.y(), m_rcView.bottom() - rcBound.height()));
        return pos;
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        if (ThumbnailScene* pScene = qobject_cast<ThumbnailScene*>(scene()))
        {
            pScene->onNavigatorPosChanged();
        }
    }
    return _base::itemChange(change, value);
}


ThumbnailScene::ThumbnailScene(QRectF sceneRect, QObject* parent)
    : _base(sceneRect, parent)
    , m_origin(nullptr)
    , m_scaleX(1.)
    , m_scaleY(1.)
    , m_navigator(nullptr)
{
}

ThumbnailScene::~ThumbnailScene()
{
}

void ThumbnailScene::initNavigator()
{
    QRectF rc = this->sceneRect();
    qreal W = rc.width();
    qreal H = rc.height();
    m_navigator = new NavigatorItem(rc, 0, 0, W / 2., H / 2.);
    m_navigator->setPos(rc.center() - QPointF(W / 2., H / 2.));

    addItem(m_navigator);
}

void ThumbnailScene::initScene(ZenoSubGraphScene* pScene)
{
    clear();
    if (m_origin) {
        disconnect(pScene, &ZenoSubGraphScene::nodePosChanged, this, &ThumbnailScene::onNodePosChanged);
        disconnect(pScene, &QGraphicsScene::sceneRectChanged, this, &ThumbnailScene::onSceneRectChanged);
        m_origin = nullptr;
    }

    m_navigator = nullptr;
    m_nodes.clear();
    //case1: 源scene里面有增删查改，如果整体boudingrect没变化，则只需更新个别节点缩略图的位置。
    connect(pScene, &ZenoSubGraphScene::nodePosChanged, this, &ThumbnailScene::onNodePosChanged);
    //case2: 如果整个sceneRect有变化（比如往外拖动导致sceneRect扩大），需要更新所有节点
    connect(pScene, &ZenoSubGraphScene::sceneRectChanged, this, &ThumbnailScene::onSceneRectChanged);

    connect(pScene, &ZenoSubGraphScene::nodeInserted, this, &ThumbnailScene::onNodeInserted);
    connect(pScene, &ZenoSubGraphScene::nodeAboutToRemoved, this, &ThumbnailScene::onNodeAboutToRemoved);

    m_origin = pScene;
    initNavigator();
    onSceneRectChanged(QRectF());
}

void ThumbnailScene::onSceneRectChanged(const QRectF& rcNew)
{
    QRectF rcScene = this->sceneRect();
    QRectF rcOrigin = m_origin->sceneRect();
    QRectF rcOrigin2 = m_origin->itemsBoundingRect();

    if (rcOrigin.width() == 0 || rcOrigin.height() == 0)
        return;

    m_scaleX = rcScene.width() / rcOrigin.width();
    m_scaleY = rcScene.height() / rcOrigin.height();

    QPointF ltOffset = rcOrigin.topLeft();

    QTransform trans1 = QTransform();
    trans1.translate(-ltOffset.x(), -ltOffset.y());

    QTransform trans2 = QTransform();
    trans2.scale(m_scaleX, m_scaleY);

    m_trans = trans1 * trans2;

    ZenoNodeBase* pLeftestNode = nullptr;
    qreal minx = 1000000;

    for (auto pNode : m_origin->getNodesItem())
    {
        QRectF br = pNode->boundingRect();
        qreal W = br.width();
        qreal H = br.height();
        W = qMin(10., W * m_scaleX);
        H = qMin(6., H * m_scaleY);

        qreal posx = pNode->x();
        qreal posy = pNode->y();

        if (posx < minx) {
            pLeftestNode = pNode;
            minx = posx;
        }

        QGraphicsRectItem* pThumbNode = nullptr;
        const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
        auto iter = m_nodes.find(path);
        if (iter == m_nodes.end())
        {
            pThumbNode = onNewThumbNode(pNode, path);
        }
        else {
            pThumbNode = iter.value();
        }

        QPointF pt = m_trans.map(pNode->pos());
        pThumbNode->setPos(pt);
    }
    if (pLeftestNode) {
        pLeftestNode->pos();
    }
}

QGraphicsRectItem* ThumbnailScene::onNewThumbNode(const ZenoNodeBase* pNode, const zeno::ObjPath& path)
{
    QRectF br = pNode->boundingRect();
    qreal W = br.width();
    qreal H = br.height();
    W = qMin(10., W * m_trans.m11());
    H = qMin(6., H * m_trans.m22());

    QGraphicsRectItem* pThumbNode = new QGraphicsRectItem(0, 0, W, H);
    pThumbNode->setBrush(QColor("#989898"));
    m_nodes.insert(path, pThumbNode);
    addItem(pThumbNode);
    return pThumbNode;
}

void ThumbnailScene::onNodePosChanged(const ZenoNodeBase* pNode)
{
    const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
    auto iter = m_nodes.find(path);
    ZASSERT_EXIT(iter != m_nodes.end());
    QGraphicsRectItem* pThumbNode = iter.value();

    QPointF pt = m_trans.map(pNode->pos());
    pThumbNode->setPos(pt);
}

void ThumbnailScene::onNodeInserted(const ZenoNodeBase* pNode)
{
    const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
    QGraphicsRectItem* pThumbNode = onNewThumbNode(pNode, path);
    QPointF pt = m_trans.map(pNode->pos());
    pThumbNode->setPos(pt);
}

void ThumbnailScene::onNodeAboutToRemoved(const ZenoNodeBase* pNode)
{
    const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
    auto iter = m_nodes.find(path);
    ZASSERT_EXIT(iter != m_nodes.end());
    QGraphicsRectItem* pThumbNode = iter.value();
    m_nodes.remove(path);
    removeItem(pThumbNode);
}

void ThumbnailScene::onNavigatorPosChanged()
{
    QRectF rcNav = QRectF(m_navigator->pos(), m_navigator->rect().size());
    QRectF sceneRc = this->sceneRect();
    emit navigatorChanged(rcNav, sceneRc);
}

ZenoSubGraphScene* ThumbnailScene::originalScene() const {
    return m_origin;
}

void ThumbnailScene::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
}

void ThumbnailScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseMoveEvent(event);
}

void ThumbnailScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
}

void ThumbnailScene::keyPressEvent(QKeyEvent* event)
{
    _base::keyPressEvent(event);
}

void ThumbnailScene::keyReleaseEvent(QKeyEvent* event)
{
    _base::keyReleaseEvent(event);
}

void ThumbnailScene::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{

}

void ThumbnailScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
    _base::wheelEvent(event);
    int wtf = event->delta();
    if (wtf > 0)
    {
        m_navigator->resize(true);
    }
    else
    {
        m_navigator->resize(false);
    }
}

void ThumbnailScene::focusOutEvent(QFocusEvent* event)
{
    _base::focusOutEvent(event);
}