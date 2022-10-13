#ifndef __ZSOCKET_LAYOUT_H__
#define __ZSOCKET_LAYOUT_H__

#include "zgraphicslayout.h"
#include "zgraphicstextitem.h"

class ZenoSocketItem;
class ZSimpleTextItem;
class ZSocketGroupItem;

class ZSocketLayout : public ZGraphicsLayout
{
public:
    ZSocketLayout(
            const QString& sockName,
            bool bInput,
            bool editable,
            Callback_OnClick cbSock,
            Callback_EditContentsChange cb
            );
    ~ZSocketLayout();
    QPointF getPortPos();
    void setControl(QGraphicsItem* pControl);
    void setValue(const QVariant& value);
    void updateSockName(const QString& name);
    QGraphicsItem* textItem() const;
    QGraphicsItem* control() const;
    QGraphicsWidget* widgetControl() const;
    ZenoSocketItem* socketItem() const;

private:
    QGraphicsItem* m_text;
    QGraphicsItem* m_control;
    bool m_bInput;
    bool m_bEditable;
};

#endif