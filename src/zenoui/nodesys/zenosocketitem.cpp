#include "zenosocketitem.h"

ZenoSocketItem::ZenoSocketItem(SOCKET_INFO info, const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent)
    : ZenoImageItem(elem, sz, parent)
    , m_info(info)
{
}

int ZenoSocketItem::type() const
{
    return Type;
}

SOCKET_INFO ZenoSocketItem::getSocketInfo()
{
    m_info.pos = sceneBoundingRect().center();
    return m_info;
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
