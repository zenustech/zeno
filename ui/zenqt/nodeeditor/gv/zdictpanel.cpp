#include "zdictpanel.h"
#include "zsocketlayout.h"
#include "control/renderparam.h"
#include "style/zenostyle.h"
#include "zenosocketitem.h"
#include "variantptr.h"
#include "zenoparamwidget.h"
#include "util/uihelper.h"
#include "zitemfactory.h"
#include "zenogvhelper.h"
#include <zeno/utils/scope_exit.h>
#include "model/parammodel.h"
#include "model/GraphModel.h"
#include "reflect/reflection.generated.hpp"


class ZDictPanel;

class ZDictItemLayout : public ZGraphicsLayout
{
public:
    ZDictItemLayout(bool bDict, const QModelIndex& paramIdx, const QString& key, const CallbackForSocket& cbSock, ZDictPanel* panel)
        : ZGraphicsLayout(true)
        , m_key(key)
        , m_editText(nullptr)
        , m_pRemoveBtn(nullptr)
        , m_pMoveUpBtn(nullptr)
        , m_bDict(bDict)
    {
        const int cSocketWidth = ZenoStyle::dpiScaled(12);
        const int cSocketHeight = ZenoStyle::dpiScaled(12);

        m_socket = new ZenoSocketItem(paramIdx, QSizeF(cSocketWidth, cSocketHeight), true);
        const bool bInput = paramIdx.data(ROLE_ISINPUT).toBool();
        m_socket->setInnerKey(m_key);

        QObject::connect(m_socket, &ZenoSocketItem::clicked, [=](bool bInput) {
            if (cbSock.cbOnSockClicked)
                cbSock.cbOnSockClicked(m_socket);
        });

        //move up button
        ImageElement elem;
        elem.image = ":/icons/moveUp.svg";
        elem.imageHovered = ":/icons/moveUp-on.svg";
        elem.imageOn = ":/icons/moveUp.svg";
        elem.imageOnHovered = ":/icons/moveUp-on.svg";
        m_pMoveUpBtn = new ZenoImageItem(elem, ZenoStyle::dpiScaledSize(QSizeF(20, 20)));
        QObject::connect(m_pMoveUpBtn, &ZenoImageItem::clicked, [=]() {
            panel->onMoveUpBtnClicked(m_key);
        });

        //close button
        elem.image = ":/icons/closebtn.svg";
        elem.imageHovered = ":/icons/closebtn_on.svg";
        elem.imageOn = ":/icons/closebtn.svg";
        elem.imageOnHovered = ":/icons/closebtn_on.svg";
        m_pRemoveBtn = new ZenoImageItem(elem, ZenoStyle::dpiScaledSize(QSizeF(20, 20)));
        QObject::connect(m_pRemoveBtn, &ZenoImageItem::clicked, [=]() {
            panel->onRemovedBtnClicked(m_key);
        });

        Callback_EditFinished cbEditFinished = [=](QVariant newValue) {
            panel->onKeyEdited(m_key, newValue.toString());
            m_key = newValue.toString();
            m_socket->setInnerKey(m_key);
        };

        CallbackCollection cbSet;
        cbSet.cbEditFinished = cbEditFinished;

        m_editText = zenoui::createItemWidget(key, zeno::Lineedit, Param_String, cbSet, nullptr, zeno::reflect::Any());
        m_editText->setEnabled(m_bDict);
        m_editText->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));

        if (bInput)
        {
            addItem(m_socket, Qt::AlignVCenter);
            addItem(m_editText, Qt::AlignVCenter);
            addItem(m_pMoveUpBtn, Qt::AlignVCenter);
            addItem(m_pRemoveBtn, Qt::AlignVCenter);
        }
        else
        {
            addItem(m_pRemoveBtn, Qt::AlignVCenter);
            addItem(m_pMoveUpBtn, Qt::AlignVCenter);
            addItem(m_editText, Qt::AlignVCenter);
            addItem(m_socket, Qt::AlignVCenter);
        }

        setSpacing(ZenoStyle::dpiScaled(5));
    }

    ZenoSocketItem* socketItem() const
    {
        return m_socket;
    }

    QPersistentModelIndex socketIdx() const
    {
        return QModelIndex();
    }

    QString key() const {
        return m_key;
    }

    void updateName(const QString& newKeyName)
    {
        ZenoGvHelper::setValue(m_editText, Param_String, newKeyName, nullptr);
    }

    void setEnable(bool bEnable)
    {
        m_socket->setEnabled(bEnable);
        m_editText->setEnabled(m_bDict ? bEnable : false);
        m_pRemoveBtn->setEnabled(bEnable);
        m_pMoveUpBtn->setEnabled(bEnable);
    }

private:
    QString m_key;
    ZenoSocketItem* m_socket;
    QGraphicsItem* m_editText;
    ZenoImageItem* m_pRemoveBtn;
    ZenoImageItem* m_pMoveUpBtn;
    bool m_bDict;
};


ZDictPanel::ZDictPanel(ZDictSocketLayout* pLayout, const QPersistentModelIndex& paramIdx, const CallbackForSocket& cbSock)
    : ZLayoutBackground()
    , m_paramIdx(paramIdx)
    , m_pDictLayout(pLayout)
    , m_pEditBtn(nullptr)
    , m_cbSock(cbSock)
    , m_bDict(true)
{
    int radius = ZenoStyle::dpiScaled(0);
    setRadius(radius, radius, radius, radius);
    QColor clr("#24282E");
    setColors(false, clr, clr, clr);
    setBorder(0, QColor());

    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);

    pVLayout->setContentsMargin(8, 0, 8, 8);
    pVLayout->setSpacing(ZenoStyle::dpiScaled(8));

    const zeno::ParamType type = (zeno::ParamType)m_paramIdx.data(ROLE_PARAM_TYPE).toInt();

    const QString& coreType = m_paramIdx.data(ROLE_PARAM_TYPE).toString();
    m_bDict = type == Param_Dict;

    bool bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();

    zeno::ParamPrimitive paramInfo = m_paramIdx.data(ROLE_PARAM_INFO).value<zeno::ParamPrimitive>();
    for (int r = 0; r < paramInfo.links.size(); r++)
    {
        zeno::EdgeInfo edge = paramInfo.links[r];
        const QString& key = bInput ? QString::fromStdString(edge.inKey) : QString::fromStdString(edge.outKey);
        ZDictItemLayout* pkey = new ZDictItemLayout(m_bDict, m_paramIdx, key, cbSock, this);
        pVLayout->addLayout(pkey);
    }

    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(paramIdx.model());
    ParamsModel* paramsM = qobject_cast<ParamsModel*>(pModel);

    QString btnName = m_bDict ? "+ Add Key" : "+ Add Item";
    if (!bInput)
    {
        m_pEditBtn = new ZenoParamPushButton(btnName, "dictkeypanel");
        pVLayout->addItem(m_pEditBtn, Qt::AlignHCenter);
    }
    
    {
        connect(paramsM, &ParamsModel::linkAboutToBeInserted, this, [=](const zeno::EdgeInfo& link) {
            ZDictItemLayout* pkey = new ZDictItemLayout(m_bDict, 
                m_paramIdx, QString::fromStdString(link.inKey), cbSock, this);
            int lastItemPos = bInput ? pVLayout->count() : pVLayout->count() - 1;
            m_layout->addLayout(pkey);
            ZGraphicsLayout::updateHierarchy(pVLayout);
        });

        connect(paramsM, &ParamsModel::linkAboutToBeRemoved, this, [=](const zeno::EdgeInfo& link) {
            removeKey(QString::fromStdString(link.inKey));
        });
    }

    connect(m_pEditBtn, &ZenoParamPushButton::clicked, this, [=]() {
        QString newKey = generateNewKey();
        ZDictItemLayout* pkey = new ZDictItemLayout(m_bDict, m_paramIdx, newKey, cbSock, this);
        int lastItemPos = bInput ? pVLayout->count() : pVLayout->count() - 1;
        m_layout->insertLayout(lastItemPos, pkey);
        ZGraphicsLayout::updateHierarchy(pVLayout);
    });

    setLayout(pVLayout);
}

ZDictPanel::~ZDictPanel()
{
}

void ZDictPanel::setEnable(bool bEnable)
{
    m_pEditBtn->setEnabled(bEnable);
    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            pDictItem->setEnable(bEnable);
        }
    }
}

QSet<QString> ZDictPanel::keyNames() const
{
    QSet<QString> names;
    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            names.insert(pDictItem->key());
        }
    }
    return names;
}

QString ZDictPanel::generateNewKey() const
{
    QString key = "obj";
    QSet<QString> names = keyNames();
    int i = 0;
    QString newName = QString("obj%1").arg(i);
    while (names.find(newName) != names.end()) {
        i++;
        newName = QString("obj%1").arg(i);
    }
    return newName;
}

void ZDictPanel::removeKey(const QString& keyName)
{
    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            if (pDictItem && pDictItem->key() == keyName)
            {
                m_layout->removeElement(i);
                ZGraphicsLayout::updateHierarchy(m_layout);
                return;
            }
        }
    }
}

void ZDictPanel::onRemovedBtnClicked(const QString& keyName)
{
    bool bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();
    PARAM_LINKS links = m_paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_paramIdx.model());
    ParamsModel* paramsM = qobject_cast<ParamsModel*>(pModel);

    for (auto linkIdx : links) {
        zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        if (bInput && edge.inKey == keyName.toStdString()) {
            paramsM->getGraph()->removeLink(edge);
            removeKey(keyName);
            break;
        }
        else if (!bInput && edge.outKey == keyName.toStdString()) {
            paramsM->getGraph()->removeLink(edge);
            removeKey(keyName);
            break;
        }
    }

    if (m_cbSock.cbOnSockLayoutChanged)
        m_cbSock.cbOnSockLayoutChanged();
}

void ZDictPanel::onMoveUpBtnClicked(const QString& keyName)
{
    bool bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_paramIdx.model());
    ParamsModel* paramsM = qobject_cast<ParamsModel*>(pModel);

    GraphModel* pGraphM = paramsM->getGraph();  //may be a assets.
    if (!pGraphM)
        return;

    PARAM_LINKS links = m_paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
    for (auto linkIdx : links) {
        zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        if (bInput && edge.inKey == keyName.toStdString()) {
            pGraphM->moveUpLinkKey(linkIdx, bInput, edge.inKey);
            break;
        }
        else if (!bInput && edge.outKey == keyName.toStdString()) {
            pGraphM->moveUpLinkKey(linkIdx, bInput, edge.outKey);
            break;
        }
    }

    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            if (pDictItem && pDictItem->key() == keyName)
            {
                m_layout->moveItem(i, i - 1);
                ZGraphicsLayout::updateHierarchy(m_layout);
                break;
            }
        }
    }

    if (m_cbSock.cbOnSockLayoutChanged)
        m_cbSock.cbOnSockLayoutChanged();
}

void ZDictPanel::onKeyEdited(const QString& oldKey, const QString& newKey)
{
    bool bInput = m_paramIdx.data(ROLE_ISINPUT).toBool();
    PARAM_LINKS links = m_paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_paramIdx.model());
    ParamsModel* paramsM = qobject_cast<ParamsModel*>(pModel);

    for (auto linkIdx : links) {
        zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
        if (bInput && edge.inKey == oldKey.toStdString()) {
            paramsM->getGraph()->updateLink(linkIdx, bInput, oldKey, newKey);
            break;
        }
        else if (!bInput && edge.outKey == oldKey.toStdString()) {
            paramsM->getGraph()->updateLink(linkIdx, bInput, oldKey, newKey);
            break;
        }
    }
}

ZenoSocketItem* ZDictPanel::socketItemByIdx(const QModelIndex& sockIdx, const QString keyName) const
{
    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            if (pDictItem && pDictItem->key() == keyName)
            {
                return pDictItem->socketItem();
            }
        }
    }
    return nullptr;
}
