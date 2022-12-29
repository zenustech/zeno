#include "zsocketlayout.h"
#include "zenosocketitem.h"
#include "zlayoutbackground.h"
#include "zgraphicstextitem.h"
#include "zassert.h"
#include "style/zenostyle.h"
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zdictpanel.h"
#include "variantptr.h"


ZSocketLayout::ZSocketLayout(
        IGraphsModel* pModel,
        const QPersistentModelIndex& viewSockIdx,
        bool bInput
    )
    : ZGraphicsLayout(true)
    , m_text(nullptr)
    , m_control(nullptr)
    , m_bInput(bInput)
    , m_bEditable(false)
    , m_viewSockIdx(viewSockIdx)
{
}

ZSocketLayout::~ZSocketLayout()
{
}

void ZSocketLayout::initUI(IGraphsModel* pModel, const CallbackForSocket& cbSock)
{
    QString sockName;
    int sockProp = 0;
    if (!m_viewSockIdx.isValid())
    {
        //test case.
        sockName = "test";
    }
    else
    {
        sockName = m_viewSockIdx.data(ROLE_VPARAM_NAME).toString();
        sockProp = m_viewSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
        m_bEditable = sockProp & SOCKPROP_EDITABLE;
    }

    if (m_bEditable)
    {
        Callback_EditContentsChange cbFuncRenameSock = [=](QString oldText, QString newText) {
            pModel->ModelSetData(m_viewSockIdx, newText, ROLE_PARAM_NAME);
        };
        m_text = new ZSocketEditableItem(m_viewSockIdx, sockName, m_bInput, cbSock.cbOnSockClicked, cbFuncRenameSock);
        addItem(m_text, m_bInput ? Qt::AlignVCenter : Qt::AlignRight | Qt::AlignVCenter);
        setSpacing(ZenoStyle::dpiScaled(32));
    }
    else
    {
        m_text = new ZSocketGroupItem(m_viewSockIdx, sockName, m_bInput, cbSock.cbOnSockClicked);
        addItem(m_text, m_bInput ? Qt::AlignVCenter : Qt::AlignRight | Qt::AlignVCenter);
        setSpacing(ZenoStyle::dpiScaled(32));
    }
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

QPointF ZSocketLayout::getSocketPos(const QModelIndex& sockIdx, bool& exist)
{
    exist = false;
    if (m_viewSockIdx == sockIdx || m_viewSockIdx.data(ROLE_PARAM_COREIDX).toModelIndex() == sockIdx)
    {
        int sockProp = m_viewSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
        if (sockProp & SOCKPROP_EDITABLE)
        {
            ZSocketEditableItem* pEdit = static_cast<ZSocketEditableItem*>(m_text);
            exist = true;
            return pEdit->socketItem()->center();
        }
        else
        {
            ZSocketGroupItem* pEdit = static_cast<ZSocketGroupItem*>(m_text);
            exist = true;
            return pEdit->socketItem()->center();
        }
    }
    return QPointF();
}

ZenoSocketItem* ZSocketLayout::socketItemByIdx(const QModelIndex& sockIdx) const
{
    if (m_viewSockIdx == sockIdx || m_viewSockIdx.data(ROLE_PARAM_COREIDX).toModelIndex() == sockIdx)
    {
        int sockProp = m_viewSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
        if (sockProp & SOCKPROP_EDITABLE)
        {
            ZSocketEditableItem* pEdit = static_cast<ZSocketEditableItem*>(m_text);
            return pEdit->socketItem();
        }
        else
        {
            ZSocketGroupItem* pEdit = static_cast<ZSocketGroupItem*>(m_text);
            return pEdit->socketItem();
        }
    }
    return nullptr;
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


////////////////////////////////////////////////////////////////////////////////////
ZDictSocketLayout::ZDictSocketLayout(
        IGraphsModel* pModel,
        const QPersistentModelIndex& viewSockIdx,
        bool bInput
    )
    : ZSocketLayout(pModel, viewSockIdx, bInput)
    , m_panel(nullptr)
{
}

ZDictSocketLayout::~ZDictSocketLayout()
{
}

void ZDictSocketLayout::initUI(IGraphsModel* pModel, const CallbackForSocket& cbSock)
{
    setHorizontal(false);

    PARAM_CLASS sockCls = (PARAM_CLASS)m_viewSockIdx.data(ROLE_PARAM_CLASS).toInt();
    bool bInput = sockCls == PARAM_INPUT || sockCls == PARAM_INNER_INPUT;
    const QString& sockName = m_viewSockIdx.data(ROLE_VPARAM_NAME).toString();
    m_text = new ZSocketGroupItem(m_viewSockIdx, sockName, m_bInput, cbSock.cbOnSockClicked);

    QSizeF iconSz = ZenoStyle::dpiScaledSize(QSizeF(28, 28));
    m_collaspeBtn = new ZenoImageItem(":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_unfold.svg", iconSz);
    m_collaspeBtn->setCheckable(true);

    m_panel = new ZDictPanel(this, m_viewSockIdx, cbSock);

    ZGraphicsLayout *pHLayout = new ZGraphicsLayout(true);
    ZGraphicsLayout *pHPanelLayout = new ZGraphicsLayout(true);
    pHLayout->setDebugName("dict socket");

    if (bInput)
    {
        pHLayout->addItem(m_text);
        pHLayout->addItem(m_collaspeBtn);

        pHPanelLayout->addItem(m_panel);
        pHPanelLayout->addSpacing(ZenoStyle::dpiScaled(64), QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred));
    }
    else
    {
        pHLayout->addSpacing(-1, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred));
        pHLayout->addItem(m_collaspeBtn);
        pHLayout->addItem(m_text);

        pHPanelLayout->addSpacing(ZenoStyle::dpiScaled(64), QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred));
        pHPanelLayout->addItem(m_panel);
    }

    addLayout(pHLayout);
    addLayout(pHPanelLayout);

    setSpacing(ZenoStyle::dpiScaled(0));

    QAbstractItemModel *dictkeyModel = QVariantPtr<QAbstractItemModel>::asPtr(m_viewSockIdx.data(ROLE_VPARAM_LINK_MODEL));
    ZASSERT_EXIT(dictkeyModel);
    bool bCollasped = dictkeyModel->data(QModelIndex(), ROLE_COLLASPED).toBool();
    m_collaspeBtn->toggle(!bCollasped);
    m_panel->setVisible(!bCollasped);

    QObject::connect(m_collaspeBtn, &ZenoImageItem::toggled, [=](bool bChecked) {

        QAbstractItemModel *keyModel = QVariantPtr<QAbstractItemModel>::asPtr(m_viewSockIdx.data(ROLE_VPARAM_LINK_MODEL));
        if (keyModel)
            keyModel->setData(QModelIndex(), !bChecked, ROLE_COLLASPED);

        m_panel->setVisible(bChecked);
        ZGraphicsLayout::updateHierarchy(pHLayout);
        if (cbSock.cbOnSockLayoutChanged)
            cbSock.cbOnSockLayoutChanged();
    });
}

void ZDictSocketLayout::setCollasped(bool bCollasped)
{
    m_collaspeBtn->toggle(!bCollasped);
}

ZenoSocketItem* ZDictSocketLayout::socketItemByIdx(const QModelIndex& sockIdx) const
{
    // dict/list socket match?
    if (ZenoSocketItem *pItem = ZSocketLayout::socketItemByIdx(sockIdx))
        return pItem;

    ZASSERT_EXIT(m_panel, nullptr);
    //and then search items on panel.
    return m_panel->socketItemByIdx(sockIdx);
}

QPointF ZDictSocketLayout::getSocketPos(const QModelIndex& sockIdx, bool& exist)
{
    if (m_viewSockIdx == sockIdx)
    {
        ZSocketGroupItem *pEdit = static_cast<ZSocketGroupItem *>(m_text);
        exist = true;
        return pEdit->socketItem()->center();
    }

    if (m_panel->isVisible())
    {
        if (ZenoSocketItem* pSocketItem = m_panel->socketItemByIdx(sockIdx))
        {
            exist = true; 
            return pSocketItem->center();
        }
    }
    else
    {
        ZSocketGroupItem *pEdit = static_cast<ZSocketGroupItem *>(m_text);
        exist = true;
        return pEdit->socketItem()->center();
    }
}
