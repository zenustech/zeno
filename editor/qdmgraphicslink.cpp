#include "qdmgraphicslink.h"
#include <QGraphicsScene>

ZENO_NAMESPACE_BEGIN

QDMGraphicsLink::QDMGraphicsLink()
{
}

QRectF QDMGraphicsLink::boundingRect() const
{
    return shape().boundingRect();
}

QPainterPath QDMGraphicsLink::shape() const
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

void QDMGraphicsLink::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    QPen pen;
    pen.setColor(QColor(isSelected() ? 0xff8800 : 0x44aacc));
    pen.setWidthF(WIDTH);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(shape());
}

ZENO_NAMESPACE_END
