#ifndef __ZDICT_PANEL_H__
#define __ZDICT_PANEL_H__

#include "zlayoutbackground.h"
#include "callbackdef.h"

class ZDictSocketLayout;

class ZDictPanel : public ZLayoutBackground
{
    Q_OBJECT
public:
    ZDictPanel(ZDictSocketLayout* pLayout, const QPersistentModelIndex& viewSockIdx, const CallbackForSocket& cbSock);
    ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx) const;

private:
    const QPersistentModelIndex m_viewSockIdx;
    ZDictSocketLayout* m_pDictLayout;
};

#endif