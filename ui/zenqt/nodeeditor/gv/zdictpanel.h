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
    ~ZDictPanel();
    ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx, const QString keyName) const;

    void onRemovedBtnClicked(const QString& keyName);
    void onMoveUpBtnClicked(const QString& keyName);
    void onKeyEdited(const QString& oldKey, const QString& newKey);

private:
    void setEnable(bool bEnable);
    QSet<QString> keyNames() const;
    QString generateNewKey() const;
    void removeKey(const QString& key);

    const QPersistentModelIndex m_paramIdx;
    CallbackForSocket m_cbSock;
    ZDictSocketLayout* m_pDictLayout;
    ZenoParamPushButton* m_pEditBtn;
    bool m_bDict;
};

#endif