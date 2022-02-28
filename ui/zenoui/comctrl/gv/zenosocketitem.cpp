#include "zenosocketitem.h"

ZenoSocketItem::ZenoSocketItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent)
    : ZenoImageItem(elem, sz, parent)
{
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
