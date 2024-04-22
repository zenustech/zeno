#ifndef __SOCKET_BACKGROUND_ITEM_H__
#define __SOCKET_BACKGROUND_ITEM_H__

#include "zlayoutbackground.h"

class ZenoSocketItem;

class SocketBackgroud : public ZLayoutBackground
{
    Q_OBJECT
public:
    SocketBackgroud(bool bInput, bool bPanelLayout, QGraphicsItem* parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    void setSocketItem(ZenoSocketItem* pSocket);

private slots:
    void onGeometryChanged();

private:
    ZenoSocketItem* m_socket;
    bool m_bInput;
    bool m_bPanelLayout;
};


#endif