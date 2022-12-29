#ifndef __ZDICT_PANEL_H__
#define __ZDICT_PANEL_H__

#include "zlayoutbackground.h"
#include "callbackdef.h"

class ZDictSocketLayout;
class ZenoParamPushButton;

class ZDictPanel : public ZLayoutBackground
{
    Q_OBJECT
public:
    ZDictPanel(ZDictSocketLayout* pLayout, const QPersistentModelIndex& viewSockIdx, const CallbackForSocket& cbSock);
    ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx) const;

private:
    void setEnable(bool bEnable);

    const QPersistentModelIndex m_viewSockIdx;
    ZDictSocketLayout* m_pDictLayout;
    ZenoParamPushButton* m_pEditBtn;
};

#endif