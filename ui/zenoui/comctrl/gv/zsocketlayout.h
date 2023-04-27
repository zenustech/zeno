#ifndef __ZSOCKET_LAYOUT_H__
#define __ZSOCKET_LAYOUT_H__

#include "zgraphicslayout.h"
#include "zgraphicstextitem.h"
#include "callbackdef.h"

class ZenoSocketItem;
class ZSimpleTextItem;
class ZSocketPlainTextItem;
class ZDictPanel;
class ZenoImageItem;
class IGraphsModel;
class ZenoParamGroupLine;

class ZSocketLayout : public ZGraphicsLayout
{
public:
    ZSocketLayout(
            IGraphsModel* pModel,
            const QPersistentModelIndex& viewSockIdx,
            bool bInput
            );
    ~ZSocketLayout();
    virtual void initUI(IGraphsModel* pModel, const CallbackForSocket& cbSock);
    void setControl(QGraphicsItem* pControl);
    virtual void updateSockName(const QString &name);
    void updateSockNameToolTip(const QString &tip);
    QGraphicsItem* control() const;
    ZenoSocketItem* socketItem() const;
    virtual ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx) const;
    virtual QPointF getSocketPos(const QModelIndex& sockIdx, bool& exist);
    QPersistentModelIndex viewSocketIdx() const;

protected:
    ZenoSocketItem* m_socket;
    QGraphicsItem* m_text;
    QGraphicsItem* m_control;
    bool m_bInput;
    bool m_bEditable;
    const QPersistentModelIndex m_viewSockIdx;
};

class ZDictSocketLayout : public ZSocketLayout
{
public:
    ZDictSocketLayout(
        IGraphsModel* pModel,
        const QPersistentModelIndex& viewSockIdx,
        bool bInput
    );
    ~ZDictSocketLayout();
    void initUI(IGraphsModel* pModel, const CallbackForSocket& cbSock) override;
    ZenoSocketItem* socketItemByIdx(const QModelIndex& sockIdx) const override;
    QPointF getSocketPos(const QModelIndex& sockIdx, bool& exist) override;
    void setCollasped(bool bCollasped);

private:
    ZDictPanel* m_panel;
    ZenoImageItem* m_collaspeBtn;
};

class ZGroupSocketLayout : public ZSocketLayout {
  public:
    ZGroupSocketLayout(IGraphsModel *pModel, const QPersistentModelIndex &viewSockIdx, bool bInput);
    ~ZGroupSocketLayout();
    void initUI(IGraphsModel *pModel, const CallbackForSocket &cbSock) override;
    ZenoSocketItem *socketItemByIdx(const QModelIndex &sockIdx) const override;
    QPointF getSocketPos(const QModelIndex &sockIdx, bool &exist) override;
    void updateSockName(const QString &name) override;

  private:
    ZenoParamGroupLine *m_pGroupLine;
};



#endif