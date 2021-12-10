#include "zenolink.h"
#include "zenosubgraphscene.h"


ZenoLink::ZenoLink(QGraphicsItem *parent)
    : _base(parent)
{
}

QRectF ZenoLink::boundingRect() const
{
    return shape().boundingRect();
}

QPainterPath ZenoLink::shape() const
{
    auto src = getSrcPos();
    auto dst = getDstPos();
    if (hasLastPath && src == lastSrcPos && dst == lastSrcPos)
        return lastPath;

    QPainterPath path(src);
    if (BEZIER == 0) {
        path.lineTo(dst);
    } else {
        float dist = dst.x() - src.x();
        dist = std::clamp(std::abs(dist), 40.f, 700.f) * BEZIER;
        path.cubicTo(src.x() + dist, src.y(),
                     dst.x() - dist, dst.y(),
                     dst.x(), dst.y());
    }

    hasLastPath = true;
    lastSrcPos = src;
    lastDstPos = dst;
    lastPath = path;
    return path;
}

void ZenoLink::paint(QPainter* painter, QStyleOptionGraphicsItem const* styleOptions, QWidget* widget)
{
    painter->save();
    QPen pen;
    pen.setColor(QColor(isSelected() ? 0xff8800 : 0x44aacc));
    pen.setWidthF(WIDTH);
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(shape());
    painter->restore();
}


ZenoLinkFull::ZenoLinkFull(ZenoSubGraphScene *pScene, const QString &fromId, const QString &fromPort, const QString &toId, const QString &toPort)
    : ZenoLink(nullptr)
    , m_scene(pScene)
    , m_fromNodeid(fromId)
    , m_fromPort(fromPort)
    , m_toNodeid(toId)
    , m_toPort(toPort)
{
    setZValue(-10);
}

QPointF ZenoLinkFull::getSrcPos() const
{
    QPointF pos = m_scene->getSocketPos(false, m_fromNodeid, m_fromPort);
    return pos;
}

QPointF ZenoLinkFull::getDstPos() const
{
    QPointF pos = m_scene->getSocketPos(true, m_toNodeid, m_toPort);
    return pos;
}

void ZenoLinkFull::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ZenoLink::mousePressEvent(event);
}

