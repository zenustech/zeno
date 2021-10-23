#include "qdmgraphicslink.h"

QDMGraphicsLink::QDMGraphicsLink()
{

}

QRectF QDMGraphicsLink::boundingRect() const
{
    return QRectF();
}

void QDMGraphicsLink::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    auto src = getSrcPos();
    auto dst = getDstPos();

    QPainterPath path(src);
    if (BEZIER == 0) {
        path.lineTo(dst);
    } else {
        float dist = dst.x() - src.x();
        dist = std::max(100.f, std::abs(dist)) * BEZIER;
        path.cubicTo(src.x() + dist, src.y(),
                     dst.x() - dist, dst.y(),
                     dst.x(), dst.y());
    }

    QPen pen;
    pen.setColor(QColor(Qt::red));
    pen.setWidthF(WIDTH);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(path);
}
