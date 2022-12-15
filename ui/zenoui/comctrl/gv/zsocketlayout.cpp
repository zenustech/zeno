#include "zsocketlayout.h"
#include "zenosocketitem.h"
#include "zgraphicstextitem.h"
#include "zassert.h"
#include "style/zenostyle.h"


ZSocketLayout::ZSocketLayout(
        const QPersistentModelIndex& viewSockIdx,
        const QString& sockName,
        bool bInput,
        bool editable,
        Callback_OnSockClicked cbSock,
        Callback_EditContentsChange cb
    )
    : ZGraphicsLayout(true)
    , m_text(nullptr)
    , m_control(nullptr)
    , m_bInput(bInput)
    , m_bEditable(editable)
    , m_viewSockIdx(viewSockIdx)
{
    if (m_bEditable) {
        m_text = new ZSocketEditableItem(viewSockIdx, sockName, bInput, cbSock, cb);
    }
    else {
        m_text = new ZSocketGroupItem(viewSockIdx, sockName, bInput, cbSock);
    }
    if (m_bInput)
        addItem(m_text, Qt::AlignVCenter);
    else
        addItem(m_text, Qt::AlignRight | Qt::AlignVCenter);

    setSpacing(ZenoStyle::dpiScaled(32));
}

ZSocketLayout::~ZSocketLayout()
{

}

void ZSocketLayout::setControl(QGraphicsItem* pControl)
{
    removeItem(m_control);
    m_control = pControl;
    addItem(m_control, Qt::AlignRight);
}

QGraphicsItem* ZSocketLayout::control() const
{
    return m_control;
}

ZenoSocketItem* ZSocketLayout::socketItem() const
{
    if (m_bEditable)
    {
        //not base on qgraphicsitem_cast because we need a unify "type", see QGraphicsItem::Type.
        ZSocketEditableItem* pEdit = static_cast<ZSocketEditableItem*>(m_text);
        return pEdit->socketItem();
    }
    else
    {
        ZSocketGroupItem* pEdit = static_cast<ZSocketGroupItem*>(m_text);
        return pEdit->socketItem();
    }
}

QPersistentModelIndex ZSocketLayout::viewSocketIdx() const
{
    return m_viewSockIdx;
}

void ZSocketLayout::updateSockName(const QString& name)
{
    if (m_bEditable)
    {
        ZSocketEditableItem* pEdit = static_cast<ZSocketEditableItem*>(m_text);
        if (pEdit)
            pEdit->updateSockName(name);
    }
    else
    {
        ZSocketGroupItem* pGroup = static_cast<ZSocketGroupItem*>(m_text);
        if (pGroup)
            pGroup->setText(name);
    }
}

void ZSocketLayout::setValue(const QVariant& value)
{
    int type = m_control->type();
    switch (type)
    {
        //...
    }
}