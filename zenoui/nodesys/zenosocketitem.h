#ifndef __ZENO_SOCKET_ITEM_H__
#define __ZENO_SOCKET_ITEM_H__

#include "zenosvgitem.h"
#include "nodesys_common.h"

class ZenoSocketItem : public ZenoImageItem
{
    Q_OBJECT
    typedef ZenoImageItem _base;
public:
    ZenoSocketItem(SOCKET_INFO info, const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent = 0);
    enum { Type = ZTYPE_SOCKET };
    int type() const override;
    SOCKET_INFO getSocketInfo() const;
    void updatePos();

private:
    SOCKET_INFO m_info;
};

#endif