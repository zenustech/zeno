#include "zsocketlayout.h"
#include "zenosocketitem.h"
#include "zlayoutbackground.h"
#include "zgraphicstextitem.h"
#include "zassert.h"
#include "style/zenostyle.h"
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zdictpanel.h"


ZSocketLayout::ZSocketLayout(
        IGraphsModel* pModel,
        const QPersistentModelIndex& viewSockIdx,
        bool bInput,
        Callback_OnSockClicked cbSock
    )
    : ZGraphicsLayout(true)
    , m_text(nullptr)
    , m_control(nullptr)
    , m_bInput(bInput)
    , m_bEditable(false)
    , m_viewSockIdx(viewSockIdx)
{
    QString sockName;
    int sockProp = 0;
    if (!viewSockIdx.isValid())
    {
        //test case.
        sockName = "test";
    }
    else
    {
        sockName = viewSockIdx.data(ROLE_VPARAM_NAME).toString();
        sockProp = viewSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
        m_bEditable = sockProp & SOCKPROP_EDITABLE;
    }

    if (m_bEditable)
    {
        Callback_EditContentsChange cbFuncRenameSock = [=](QString oldText, QString newText) {
            pModel->ModelSetData(m_viewSockIdx, newText, ROLE_PARAM_NAME);
        };
        m_text = new ZSocketEditableItem(viewSockIdx, sockName, m_bInput, cbSock, cbFuncRenameSock);
        addItem(m_text, m_bInput ? Qt::AlignVCenter : Qt::AlignRight | Qt::AlignVCenter);
        setSpacing(ZenoStyle::dpiScaled(32));
    }
    else if (sockProp & SOCKPROP_DICTPANEL)
    {
        setHorizontal(false);

        ZGraphicsLayout *pHLayout = new ZGraphicsLayout(true);

        m_text = new ZSocketGroupItem(viewSockIdx, sockName, m_bInput, cbSock);
        pHLayout->addItem(m_text);
        pHLayout->setDebugName("dict socket");

        QSizeF iconSz = ZenoStyle::dpiScaledSize(QSizeF(28, 28));
        ZenoImageItem* collaspeBtn = new ZenoImageItem(":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_unfold.svg", iconSz);
        collaspeBtn->setCheckable(true);
        pHLayout->addItem(collaspeBtn);

        addLayout(pHLayout);

        ZDictPanel* panel = new ZDictPanel(viewSockIdx);
        addItem(panel);
        setSpacing(ZenoStyle::dpiScaled(0));
        panel->hide();

        QObject::connect(collaspeBtn, &ZenoImageItem::toggled, [=](bool bChecked) {
            panel->setVisible(bChecked);
            ZGraphicsLayout::updateHierarchy(pHLayout);
        });
    }
    else
    {
        m_text = new ZSocketGroupItem(viewSockIdx, sockName, m_bInput, cbSock);
        addItem(m_text, m_bInput ? Qt::AlignVCenter : Qt::AlignRight | Qt::AlignVCenter);
        setSpacing(ZenoStyle::dpiScaled(32));
    }
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
