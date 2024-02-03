#ifndef __ZDICTSOCKETLAYOUT_H__
#define __ZDICTSOCKETLAYOUT_H__

#include "zsocketlayout.h"

class ZenoImageItem;
class ZDictPanel;

class ZDictSocketLayout : public ZSocketLayout
{
public:
    ZDictSocketLayout(const QPersistentModelIndex& paramIdx, bool bInput);
    ~ZDictSocketLayout();
    void initUI(const CallbackForSocket& cbSock) override;
    ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx, const QString keyName) const override;
    QPointF getSocketPos(const QModelIndex& sockIdx, const QString keyName, bool& exist) override;
    void setCollasped(bool bCollasped);
    void setVisible(bool bVisible);

private:
    ZDictPanel* m_panel;
    ZenoImageItem* m_collaspeBtn;
};

#endif