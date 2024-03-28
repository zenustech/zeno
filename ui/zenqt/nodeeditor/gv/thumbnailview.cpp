#include "thumbnailview.h"
#include "thumbnailscene.h"
#include "zenosubgraphscene.h"
#include "zenonode.h"
#include <QtGlobal>


ThumbnailView::ThumbnailView(QWidget* parent)
    : QGraphicsView(parent)
    , m_scene(nullptr)
{
    this->resize(QSize(300, 128));

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

void ThumbnailView::resetScene(ZenoSubGraphScene* pOriginScene)
{
    if (m_scene && m_scene->originalScene() == pOriginScene)
    {
        return;
    }

    QRect viewRect = geometry();
    if (!m_scene) {
        m_scene = new ThumbnailScene(viewRect);
        setScene(m_scene);
        connect(m_scene, &ThumbnailScene::navigatorChanged, this, &ThumbnailView::navigatorChanged);
    }
    m_scene->initScene(pOriginScene);
}

void ThumbnailView::drawBackground(QPainter* painter, const QRectF& rect)
{
    _base::drawBackground(painter, rect);
}
