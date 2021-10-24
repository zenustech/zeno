#include "qdmgraphicslink.h"
#include <QGraphicsScene>

QDMGraphicsLink::QDMGraphicsLink()
{
}

QRectF QDMGraphicsLink::boundingRect() const
{
    return scene()->sceneRect();
}

QPainterPath QDMGraphicsLink::shape() const
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
