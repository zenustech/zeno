#ifndef __ZSOCKET_LAYOUT_H__
#define __ZSOCKET_LAYOUT_H__

#include "zgraphicslayout.h"
#include "zgraphicstextitem.h"

class ZenoSocketItem;
class ZSimpleTextItem;
class ZSocketGroupItem;
class IGraphsModel;

class ZSocketLayout : public ZGraphicsLayout
{
public:
    ZSocketLayout(
            IGraphsModel* pModel,
            const QPersistentModelIndex& viewSockIdx,
            bool bInput,
            Callback_OnSockClicked cbSock
            );
    ~ZSocketLayout();
    void setControl(QGraphicsItem* pControl);
    void updateSockName(const QString& name);
    QGraphicsItem* control() const;
    ZenoSocketItem* socketItem() const;
    QPersistentModelIndex viewSocketIdx() const;

private:
    QGraphicsItem* m_text;
    QGraphicsItem* m_control;
    bool m_bInput;
    bool m_bEditable;
    const QPersistentModelIndex m_viewSockIdx;
};

#endif