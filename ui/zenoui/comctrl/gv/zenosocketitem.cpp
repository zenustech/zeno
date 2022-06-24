#include "zenosocketitem.h"
#include <zenoui/style/zenostyle.h>


ZenoSocketItem::ZenoSocketItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent)
    : ZenoImageItem(elem, sz, parent)
    , m_bLeftSock(false)
{
    setCheckable(true);
}

int ZenoSocketItem::type() const
{
    return Type;
}

void ZenoSocketItem::setOffsetToName(const QPointF& offsetToName)
{
    m_offsetToName = offsetToName;
}

void ZenoSocketItem::socketNamePosition(const QPointF& nameScenePos)
{
    QPointF namePos = mapFromScene(nameScenePos);
    setPos(namePos + m_offsetToName);
}

QRectF ZenoSocketItem::boundingRect() const
{
    static int sLargeMargin = ZenoStyle::dpiScaled(20);
    static int sSmallMargin = ZenoStyle::dpiScaled(10);

    QRectF rc = ZenoImageItem::boundingRect();
    if (m_bLeftSock) {
        rc = rc.adjusted(-sLargeMargin, -sSmallMargin, sLargeMargin, sSmallMargin);
    }
    else {
        rc = rc.adjusted(-sLargeMargin, -sSmallMargin, sLargeMargin, sSmallMargin);
    }
    return rc;
}

void ZenoSocketItem::setIsInput(bool left)
{
    m_bLeftSock = left;
}

bool ZenoSocketItem::isInput() const
{
    return m_bLeftSock;
}
