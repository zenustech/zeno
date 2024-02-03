#include "zdictsocketlayout.h"
#include "zenosocketitem.h"
#include "util/uihelper.h"
#include "model/parammodel.h"
#include "model/GraphModel.h"
#include "control/renderparam.h"
#include "style/zenostyle.h"
#include "zitemfactory.h"
#include "zenogvhelper.h"
#include "zdictpanel.h"
#include "zassert.h"
#include "control/common_id.h"


ZDictSocketLayout::ZDictSocketLayout(const QPersistentModelIndex& paramIdx, bool bInput)
    : ZSocketLayout(paramIdx, bInput)
    , m_panel(nullptr)
{
}

ZDictSocketLayout::~ZDictSocketLayout()
{
}

void ZDictSocketLayout::initUI(const CallbackForSocket& cbSock)
{
    setHorizontal(false);

    bool bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();
    //TODO: refactor about zdict panel layout.
    const QString& sockName = m_paramIdx.data(ROLE_PARAM_NAME).toString();

    QSizeF szSocket(10, 20);
    m_socket = new ZenoSocketItem(m_paramIdx, ZenoStyle::dpiScaledSize(szSocket));
    m_socket->setZValue(ZVALUE_ELEMENT);
    QObject::connect(m_socket, &ZenoSocketItem::clicked, [=](bool bInput, Qt::MouseButton button) {
        cbSock.cbOnSockClicked(m_socket, button);
        });
    QObject::connect(m_socket, &ZenoSocketItem::netLabelClicked, [=]() {
        if (cbSock.cbOnSockNetlabelClicked)
            cbSock.cbOnSockNetlabelClicked(m_socket->netLabel());
        });
    QObject::connect(m_socket, &ZenoSocketItem::netLabelEditFinished, [=]() {
        if (cbSock.cbOnSockNetlabelEdited)
            cbSock.cbOnSockNetlabelEdited(m_socket);
        });
    QObject::connect(m_socket, &ZenoSocketItem::netLabelMenuActionTriggered, [=](QAction* pAction) {
        if (cbSock.cbActionTriggered)
            cbSock.cbActionTriggered(pAction, m_paramIdx);
        });

    m_text = new ZSocketPlainTextItem(m_paramIdx, sockName, m_bInput, cbSock.cbOnSockClicked);
    m_text->setToolTip(m_paramIdx.data(ROLE_PARAM_TOOLTIP).toString());
    m_text->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(0, zenoui::g_ctrlHeight)));
    m_text->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed));

    QSizeF iconSz = ZenoStyle::dpiScaledSize(QSizeF(zenoui::g_ctrlHeight, zenoui::g_ctrlHeight));
    m_collaspeBtn = new ZenoImageItem(":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_fold.svg", ":/icons/ic_parameter_unfold.svg", iconSz);
    m_collaspeBtn->setCheckable(true);

    m_panel = new ZDictPanel(this, m_paramIdx, cbSock);

    m_collaspeBtn->toggle(false);
    m_panel->setVisible(false);

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
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

    QObject::connect(m_collaspeBtn, &ZenoImageItem::toggled, [=](bool bChecked) {
        m_panel->setVisible(bChecked);
        ZGraphicsLayout::updateHierarchy(pHLayout);
        if (cbSock.cbOnSockLayoutChanged)
            cbSock.cbOnSockLayoutChanged();
        });

    addLayout(pHLayout);
    addItem(m_panel);

    setSpacing(ZenoStyle::dpiScaled(5));
}

void ZDictSocketLayout::setCollasped(bool bCollasped)
{
    m_collaspeBtn->toggle(!bCollasped);
}

void ZDictSocketLayout::setVisible(bool bVisible)
{
    m_text->setVisible(bVisible);
    m_collaspeBtn->setVisible(bVisible);
}

ZenoSocketItem* ZDictSocketLayout::socketItemByIdx(const QModelIndex& sockIdx, const QString keyName) const
{
    if (!m_panel->isVisible())
    {
        return m_socket;
    }
    //if (ZenoSocketItem *pItem = ZSocketLayout::socketItemByIdx(sockIdx, keyName))
    //    return pItem;
    ZASSERT_EXIT(m_panel, nullptr);
    //and then search items on panel.
    return m_panel->socketItemByIdx(sockIdx, keyName);
}

QPointF ZDictSocketLayout::getSocketPos(const QModelIndex& sockIdx, const QString keyName, bool& exist)
{
    if (m_paramIdx == sockIdx)
    {
        if (ZenoSocketItem* pSocketItem = m_panel->socketItemByIdx(sockIdx, keyName))
        {
            if (m_panel->isVisible())
            {
                exist = true;
                return pSocketItem->center();
            }
            else
            {
                ZSocketPlainTextItem* pEdit = static_cast<ZSocketPlainTextItem*>(m_text);
                exist = true;
                return m_socket->center();
            }
        }
        else {
            ZSocketPlainTextItem* pEdit = static_cast<ZSocketPlainTextItem*>(m_text);
            exist = true;
            return m_socket->center();
        }
    }
    exist = false;
    return QPointF();
}