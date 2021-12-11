#include "zenolink.h"
#include "zenosubgraphscene.h"
#include "nodesys_common.h"


ZenoLink::ZenoLink(QGraphicsItem *parent)
    : _base(parent)
{
}

ZenoLink::~ZenoLink()
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

int ZenoLink::type() const
{
    return Type;
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


ZenoLinkFull::ZenoLinkFull(const EdgeInfo& info)
    : ZenoLink(nullptr)
    , m_linkInfo(info)
{
    setZValue(-10);
    setFlag(QGraphicsItem::ItemIsSelectable);
}

void ZenoLinkFull::updatePos(const QPointF& srcPos, const QPointF& dstPos)
{
    m_srcPos = srcPos;
    m_dstPos = dstPos;
}

void ZenoLinkFull::updateLink(const EdgeInfo& info)
{
    m_linkInfo = info;
}

EdgeInfo ZenoLinkFull::linkInfo() const
{
    return m_linkInfo;
}

QPointF ZenoLinkFull::getSrcPos() const
{
    return m_srcPos;
}

QPointF ZenoLinkFull::getDstPos() const
{
    return m_dstPos;
}

void ZenoLinkFull::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    ZenoLink::mousePressEvent(event);
}

int ZenoLinkFull::type() const
{
    return Type;
}
