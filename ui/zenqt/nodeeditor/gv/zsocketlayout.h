#ifndef __ZSOCKET_LAYOUT_H__
#define __ZSOCKET_LAYOUT_H__

#include "zgraphicslayout.h"
#include "zgraphicstextitem.h"
#include "callbackdef.h"

class ZenoSocketItem;
class ZSimpleTextItem;
class ZSocketPlainTextItem;
class ZenoParamGroupLine;

class ZSocketLayout : public ZGraphicsLayout
{
public:
    ZSocketLayout(
            const QPersistentModelIndex& viewSockIdx,
            bool bInput
            );
    ~ZSocketLayout();
    virtual void initUI(const CallbackForSocket& cbSock);
    void setControl(QGraphicsItem* pControl);
    virtual void updateSockName(const QString &name);
    void updateSockNameToolTip(const QString &tip);
    QGraphicsItem* socketText() const;
    virtual QGraphicsItem* control() const;
    ZenoSocketItem* socketItem() const;
    virtual ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx, const QString keyName) const;
    virtual QPointF getSocketPos(const QModelIndex& sockIdx, const QString keyName, bool& exist);
    QPersistentModelIndex viewSocketIdx() const;
    virtual void setVisible(bool bVisible);
    void setSocketVisible(bool bVisible);

protected:
    ZenoSocketItem* m_socket;
    QGraphicsItem* m_text;
    QGraphicsItem* m_control;
    bool m_bInput;
    bool m_bEditable;
    const QPersistentModelIndex m_paramIdx;
};

class ZGroupSocketLayout : public ZSocketLayout
{
public:
    ZGroupSocketLayout(const QPersistentModelIndex &viewSockIdx, bool bInput);
    ~ZGroupSocketLayout();
    void initUI(const CallbackForSocket &cbSock) override;
    ZenoSocketItem *socketItemByIdx(const QModelIndex &sockIdx, const QString keyName) const override;
    QPointF getSocketPos(const QModelIndex &sockIdx, const QString keyName, bool &exist) override;
    void updateSockName(const QString &name) override;
    void setVisible(bool bVisible);

private:
    ZenoParamGroupLine *m_pGroupLine;
};


#endif