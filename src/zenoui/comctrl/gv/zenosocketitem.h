#ifndef __ZENO_SOCKET_ITEM_H__
#define __ZENO_SOCKET_ITEM_H__

#include <zenoui/model/modeldata.h>
#include <zenoui/nodesys/zenosvgitem.h>
#include "../../nodesys/nodesys_common.h"

class ZenoSocketItem : public ZenoImageItem
{
    Q_OBJECT
    typedef ZenoImageItem _base;
public:
    ZenoSocketItem(SOCKET_INFO info, const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent = 0);
    enum { Type = ZTYPE_SOCKET };
    int type() const override;
    SOCKET_INFO getSocketInfo();
    void setOffsetToName(const QPointF& offsetToName);

public slots:
    void socketNamePosition(const QPointF& nameScenePos);

private:
    SOCKET_INFO m_info;
    QPointF m_offsetToName;
};

#endif