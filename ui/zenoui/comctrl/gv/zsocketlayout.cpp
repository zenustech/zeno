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
#include "render/common_id.h"


ZSocketLayout::ZSocketLayout(
        IGraphsModel* pModel,
        const QPersistentModelIndex& viewSockIdx,
        bool bInput
    )
    : ZGraphicsLayout(true)
    , m_text(nullptr)
    , m_control(nullptr)
    , m_socket(nullptr)
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

    QSizeF szSocket(10, 20);
    m_socket = new ZenoSocketItem(m_viewSockIdx, ZenoStyle::dpiScaledSize(szSocket));
    m_socket->setZValue(ZVALUE_ELEMENT);
    QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() {
        cbSock.cbOnSockClicked(m_socket);
    });

    if (m_bEditable)
    {
        Callback_EditContentsChange cbFuncRenameSock = [=](QString oldText, QString newText) {
            pModel->ModelSetData(m_viewSockIdx, newText, ROLE_PARAM_NAME);
        };
        m_text = new ZSocketEditableItem(m_viewSockIdx, sockName, m_bInput, cbSock.cbOnSockClicked, cbFuncRenameSock);
        setSpacing(ZenoStyle::dpiScaled(32));
    }
    else
    {
        m_text = new ZSocketGroupItem(m_viewSockIdx, sockName, m_bInput, cbSock.cbOnSockClicked);
        setSpacing(ZenoStyle::dpiScaled(32));
    }

    if (m_bInput)
    {
        addItem(m_socket, Qt::AlignVCenter);
        addItem(m_text, Qt::AlignVCenter);
    }
    else
    {
        addSpacing(-1, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred));
        addItem(m_text, Qt::AlignVCenter);
        addItem(m_socket, Qt::AlignVCenter);
    }

    setSpacing(ZenoStyle::dpiScaled(8));
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
    return m_socket;
    //if (m_bEditable)
    //{
    //    //not base on qgraphicsitem_cast because we need a unify "type", see QGraphicsItem::Type.
    //    ZSocketEditableItem* pEdit = static_cast<ZSocketEditableItem*>(m_text);
    //    return pEdit->socketItem();
    //}
    //else
    //{
    //    ZSocketGroupItem* pEdit = static_cast<ZSocketGroupItem*>(m_text);
    //    return pEdit->socketItem();
    //}
}

QPointF ZSocketLayout::getSocketPos(const QModelIndex& sockIdx, bool& exist)
{
    exist = false;
    if (m_viewSockIdx == sockIdx && m_socket)
    {
        exist = true;
        return m_socket->center();
    }
    return QPointF();
}

ZenoSocketItem* ZSocketLayout::socketItemByIdx(const QModelIndex& sockIdx) const
{
    if (m_viewSockIdx == sockIdx || m_viewSockIdx.data(ROLE_PARAM_COREIDX).toModelIndex() == sockIdx)
    {
        return m_socket;
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

    QSizeF szSocket(10, 20);
    m_socket = new ZenoSocketItem(m_viewSockIdx, ZenoStyle::dpiScaledSize(szSocket));
    m_socket->setZValue(ZVALUE_ELEMENT);
    QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() { cbSock.cbOnSockClicked(m_socket); });

    m_text = new ZSocketGroupItem(m_viewSockIdx, sockName, m_bInput, cbSock.cbOnSockClicked);

    QSizeF iconSz = ZenoStyle::dpiScaledSize(QSizeF(28, 28));
    m_collaspeBtn = new ZenoImageItem(":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_unfold.svg", iconSz);
    m_collaspeBtn->setCheckable(true);

    m_panel = new ZDictPanel(this, m_viewSockIdx, cbSock, pModel);

    ZGraphicsLayout *pHLayout = new ZGraphicsLayout(true);
    pHLayout->setDebugName("dict socket");
    pHLayout->setSpacing(ZenoStyle::dpiScaled(8));

    if (bInput)
    {
        pHLayout->addItem(m_socket, Qt::AlignVCenter);
        pHLayout->addItem(m_text, Qt::AlignVCenter);
        pHLayout->addItem(m_collaspeBtn, Qt::AlignVCenter);
    }
    else
    {
        pHLayout->addSpacing(-1, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred));
        pHLayout->addItem(m_collaspeBtn);
        pHLayout->addItem(m_text);
        pHLayout->addItem(m_socket);
    }

    addLayout(pHLayout);
    addItem(m_panel);

    setSpacing(ZenoStyle::dpiScaled(5));

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
        return m_socket->center();
    }
    if (ZenoSocketItem* pSocketItem = m_panel->socketItemByIdx(sockIdx))
    {
        if (m_panel->isVisible())
        {
            exist = true;
            return pSocketItem->center();
        }
        else
        {
            ZSocketGroupItem *pEdit = static_cast<ZSocketGroupItem*>(m_text);
            exist = true;
            return m_socket->center();
        }
    }
    exist = false;
    return QPointF();
}
