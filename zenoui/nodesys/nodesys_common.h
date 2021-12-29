#ifndef __NODESYS_COMMON_H__
#define __NODESYS_COMMON_H__

#include "../model/modeldata.h"

enum ZenoGVItemType {
    ZTYPE_NODE = QGraphicsItem::UserType + 1,
    ZTYPE_LINK,
    ZTYPE_FULLLINK,
    ZTYPE_TEMPLINK,
    ZTYPE_SOCKET,
    ZTYPE_IMAGE,
    ZTYPE_PARAMWIDGET,
    ZTYPE_NODRAGITEM,
    ZTYPE_COLOR_CHANNEL,
    ZTYPE_COLOR_RAMP,
    ZTYPE_HEATMAP,
};

enum MIME_DATA_TYPE {
    MINETYPE_MULTI_NODES = 2099,
};

struct NODES_MIME_DATA : public QObjectUserData
{
    QList<NODE_DATA> m_vecNodes;
};


#endif