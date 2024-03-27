#include "thumbnailscene.h"
#include "zenosubgraphscene.h"
#include "model/GraphModel.h"
#include "zenonode.h"
#include "zassert.h"


ThumbnailScene::ThumbnailScene(QRectF sceneRect, QObject* parent)
    : _base(sceneRect, parent)
    , m_origin(nullptr)
    , m_scaleX(1.)
    , m_scaleY(1.)
{
}

ThumbnailScene::~ThumbnailScene()
{
}

void ThumbnailScene::initScene(ZenoSubGraphScene* pScene)
{
    clear();
    if (m_origin) {
        disconnect(pScene, &ZenoSubGraphScene::nodePosChanged, this, &ThumbnailScene::onNodePosChanged);
        disconnect(pScene, &QGraphicsScene::sceneRectChanged, this, &ThumbnailScene::onSceneRectChanged);
        m_origin = nullptr;
    }

    m_nodes.clear();
    //case1: 源scene里面有增删查改，如果整体boudingrect没变化，则只需更新个别节点缩略图的位置。
    connect(pScene, &ZenoSubGraphScene::nodePosChanged, this, &ThumbnailScene::onNodePosChanged);
    //case2: 如果整个sceneRect有变化（比如往外拖动导致sceneRect扩大），需要更新所有节点
    connect(pScene, &ZenoSubGraphScene::sceneRectChanged, this, &ThumbnailScene::onSceneRectChanged);

    connect(pScene, &ZenoSubGraphScene::nodeInserted, this, &ThumbnailScene::onNodeInserted);
    connect(pScene, &ZenoSubGraphScene::nodeAboutToRemoved, this, &ThumbnailScene::onNodeAboutToRemoved);

    m_origin = pScene;
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

    ZenoNode* pLeftestNode = nullptr;
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

QGraphicsRectItem* ThumbnailScene::onNewThumbNode(const ZenoNode* pNode, const zeno::ObjPath& path)
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

void ThumbnailScene::onNodePosChanged(const ZenoNode* pNode)
{
    const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
    auto iter = m_nodes.find(path);
    ZASSERT_EXIT(iter != m_nodes.end());
    QGraphicsRectItem* pThumbNode = iter.value();

    QPointF pt = m_trans.map(pNode->pos());
    pThumbNode->setPos(pt);
}

void ThumbnailScene::onNodeInserted(const ZenoNode* pNode)
{
    const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
    QGraphicsRectItem* pThumbNode = onNewThumbNode(pNode, path);
    QPointF pt = m_trans.map(pNode->pos());
    pThumbNode->setPos(pt);
}

void ThumbnailScene::onNodeAboutToRemoved(const ZenoNode* pNode)
{
    const zeno::ObjPath& path = pNode->index().data(ROLE_NODE_UUID_PATH).value<zeno::ObjPath>();
    auto iter = m_nodes.find(path);
    ZASSERT_EXIT(iter != m_nodes.end());
    QGraphicsRectItem* pThumbNode = iter.value();
    m_nodes.remove(path);
    removeItem(pThumbNode);
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

void ThumbnailScene::focusOutEvent(QFocusEvent* event)
{
    _base::focusOutEvent(event);
}