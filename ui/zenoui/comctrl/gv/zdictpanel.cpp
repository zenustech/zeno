#include "zdictpanel.h"
#include "zsocketlayout.h"
#include "../../render/renderparam.h"
#include "../../style/zenostyle.h"
#include "zenosocketitem.h"
#include "variantptr.h"
#include <zenomodel/include/modelrole.h>
#include "zenoparamwidget.h"
#include <zenomodel/include/uihelper.h>
#include "zitemfactory.h"
#include "zenogvhelper.h"
#include <zenomodel/include/command.h>
#include <zeno/utils/scope_exit.h>

class ZDictPanel;

class ZDictItemLayout : public ZGraphicsLayout
{
public:
    ZDictItemLayout(bool bDict, const QModelIndex& keyIdx, const CallbackForSocket& cbSock, ZDictPanel* panel)
        : ZGraphicsLayout(true)
        , m_sockKeyIdx(keyIdx)
        , m_editText(nullptr)
        , m_pRemoveBtn(nullptr)
        , m_pMoveUpBtn(nullptr)
        , m_bDict(bDict)
    {
        const int cSocketWidth = ZenoStyle::dpiScaled(12);
        const int cSocketHeight = ZenoStyle::dpiScaled(12);

        PARAM_CLASS sockCls = (PARAM_CLASS)m_sockKeyIdx.data(ROLE_PARAM_CLASS).toInt();
        const bool bInput = sockCls == PARAM_INPUT || sockCls == PARAM_INNER_INPUT;

        m_socket = new ZenoSocketItem(m_sockKeyIdx, QSizeF(cSocketWidth, cSocketHeight));

        QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() {
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
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_sockKeyIdx.model());
            int r = m_sockKeyIdx.row();
            if (r > 0) {
                IGraphsModel* pGraphsModel = panel->graphsModel();
                const QString& objPath = m_sockKeyIdx.data(ROLE_OBJPATH).toString();
                pGraphsModel->addExecuteCommand(new ModelMoveCommand(pGraphsModel, objPath, r - 1));
            }
        });

        //close button
        elem.image = ":/icons/closebtn.svg";
        elem.imageHovered = ":/icons/closebtn_on.svg";
        elem.imageOn = ":/icons/closebtn.svg";
        elem.imageOnHovered = ":/icons/closebtn_on.svg";
        m_pRemoveBtn = new ZenoImageItem(elem, ZenoStyle::dpiScaledSize(QSizeF(20, 20)));
        QObject::connect(m_pRemoveBtn, &ZenoImageItem::clicked, [=]() {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_sockKeyIdx.model());
            IGraphsModel* pGraphsModel = panel->graphsModel();
            const QString& dictkeypath = panel->dictlistSocket().data(ROLE_OBJPATH).toString();
            pGraphsModel->beginTransaction("remove dict socket");
            zeno::scope_exit sp([=]() { pGraphsModel->endTransaction(); });
            pGraphsModel->addExecuteCommand(new DictKeyAddRemCommand(false, pGraphsModel, dictkeypath, m_sockKeyIdx.row()));
        });

        Callback_EditFinished cbEditFinished = [=](QVariant newValue) {
            if (newValue == m_sockKeyIdx.data().toString())
                return;
            const QString& keyObj = m_sockKeyIdx.data(ROLE_OBJPATH).toString();
            IGraphsModel* pGraphsModel = panel->graphsModel();
            pGraphsModel->addExecuteCommand(new RenameObjCommand(pGraphsModel, keyObj, newValue.toString()));
        };

        CallbackCollection cbSet;
        cbSet.cbEditFinished = cbEditFinished;

        const QString &key = m_sockKeyIdx.data().toString();
        m_editText = zenoui::createItemWidget(key, CONTROL_STRING, "string", cbSet, nullptr, QVariant());
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
        return m_sockKeyIdx;
    }
    void updateName(const QString& newKeyName)
    {
        ZenoGvHelper::setValue(m_editText, CONTROL_STRING, newKeyName, nullptr);
    }
    void setEnable(bool bEnable)
    {
        m_socket->setEnabled(bEnable);
        m_editText->setEnabled(m_bDict ? bEnable : false);
        m_pRemoveBtn->setEnabled(bEnable);
        m_pMoveUpBtn->setEnabled(bEnable);
    }

private:
    QPersistentModelIndex m_sockKeyIdx;
    ZenoSocketItem* m_socket;
    QGraphicsItem* m_editText;
    ZenoImageItem* m_pRemoveBtn;
    ZenoImageItem* m_pMoveUpBtn;
    bool m_bDict;
};


ZDictPanel::ZDictPanel(ZDictSocketLayout* pLayout, const QPersistentModelIndex& viewSockIdx, const CallbackForSocket& cbSock, IGraphsModel* pModel)
    : ZLayoutBackground()
    , m_viewSockIdx(viewSockIdx)
    , m_pDictLayout(pLayout)
    , m_pEditBtn(nullptr)
    , m_cbSock(cbSock)
    , m_bDict(true)
    , m_model(pModel)
{
    int radius = ZenoStyle::dpiScaled(0);
    setRadius(radius, radius, radius, radius);
    QColor clr("#24282E");
    setColors(false, clr, clr, clr);
    setBorder(0, QColor());

    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);

    pVLayout->setContentsMargin(8, 0, 8, 8);
    pVLayout->setSpacing(ZenoStyle::dpiScaled(8));

    const QString& coreType = m_viewSockIdx.data(ROLE_PARAM_TYPE).toString();
    m_bDict = coreType == "dict";

    QAbstractItemModel* pKeyObjModel = QVariantPtr<QAbstractItemModel>::asPtr(m_viewSockIdx.data(ROLE_VPARAM_LINK_MODEL));
    ZASSERT_EXIT(pKeyObjModel);
    for (int r = 0; r < pKeyObjModel->rowCount(); r++)
    {
        const QModelIndex& idxKey = pKeyObjModel->index(r, 0);
        QString key = idxKey.data().toString();

        ZDictItemLayout* pkey = new ZDictItemLayout(m_bDict, idxKey, cbSock, this);
        pVLayout->addLayout(pkey);
    }

    QString btnName = m_bDict ? "+ Add Key" : "+ Add Item";
    m_pEditBtn = new ZenoParamPushButton(btnName, "dictkeypanel");
    pVLayout->addItem(m_pEditBtn, Qt::AlignHCenter);

    connect(m_pEditBtn, SIGNAL(clicked()), this, SLOT(onEditBtnClicked()));
    connect(pKeyObjModel, SIGNAL(rowsInserted(const QModelIndex&, int, int)), 
        this, SLOT(onKeysInserted(const QModelIndex&, int, int)));

    connect(pKeyObjModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex &, int, int)),
        this, SLOT(onKeysAboutToBeRemoved(const QModelIndex&, int, int)));

    connect(pKeyObjModel, SIGNAL(rowsMoved(const QModelIndex&, int, int, const QModelIndex&, int)),
        this, SLOT(onKeysMoved(const QModelIndex&, int, int, const QModelIndex&, int)));

    connect(pKeyObjModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)),
        this, SLOT(onKeysModelDataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)));

    QAbstractItemModel* paramsModel = const_cast<QAbstractItemModel*>(m_viewSockIdx.model());
    connect(paramsModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)),
        this, SLOT(onAddRemoveLink(const QModelIndex&, const QModelIndex&, const QVector<int>&)));

    setLayout(pVLayout);
}

ZDictPanel::~ZDictPanel()
{
    int j;
    j = 0;
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

void ZDictPanel::onEditBtnClicked()
{
    QAbstractItemModel* pKeyObjModel = QVariantPtr<QAbstractItemModel>::asPtr(m_viewSockIdx.data(ROLE_VPARAM_LINK_MODEL));
    ZASSERT_EXIT(pKeyObjModel);
    int n = pKeyObjModel->rowCount();
    //pKeyObjModel->insertRow(n);
    const QString& dictKeyPath = m_viewSockIdx.data(ROLE_OBJPATH).toString();
    m_model->addExecuteCommand(new DictKeyAddRemCommand(true, m_model, dictKeyPath, n));
}

void ZDictPanel::onKeysAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    m_layout->removeElement(first);
    ZGraphicsLayout::updateHierarchy(m_layout);
    if (m_cbSock.cbOnSockLayoutChanged)
        m_cbSock.cbOnSockLayoutChanged();
}

void ZDictPanel::onKeysMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row)
{
    //only support move up for now.
    m_layout->moveItem(start, row);
    ZGraphicsLayout::updateHierarchy(m_layout);
    if (m_cbSock.cbOnSockLayoutChanged)
        m_cbSock.cbOnSockLayoutChanged();
}

void ZDictPanel::onKeysInserted(const QModelIndex& parent, int first, int last)
{
    //expand the dict panel anyway.
    m_pDictLayout->setCollasped(false);

    QAbstractItemModel* pKeyObjModel = QVariantPtr<QAbstractItemModel>::asPtr(m_viewSockIdx.data(ROLE_VPARAM_LINK_MODEL));
    pKeyObjModel->setData(QModelIndex(), false, ROLE_COLLASPED);

    const QModelIndex &idxKey = pKeyObjModel->index(first, 0);
    QString key = idxKey.data().toString();
    ZDictItemLayout *pkey = new ZDictItemLayout(m_bDict, idxKey, m_cbSock, this);
    m_layout->insertLayout(first, pkey);

    ZGraphicsLayout::updateHierarchy(m_layout);
    if (m_cbSock.cbOnSockLayoutChanged)
        m_cbSock.cbOnSockLayoutChanged();
}

void ZDictPanel::onKeysModelDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if ((roles.contains(Qt::DisplayRole) || roles.contains(ROLE_PARAM_NAME)) && topLeft.column() == 0)
    {
        ZDictItemLayout* pItemLayout = static_cast<ZDictItemLayout*>(m_layout->itemAt(topLeft.row())->pLayout);
        const QString &newKeyName = topLeft.data(Qt::DisplayRole).toString();
        pItemLayout->updateName(newKeyName);
    }
}

void ZDictPanel::onAddRemoveLink(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (topLeft == m_viewSockIdx)
    {
        if (roles.contains(ROLE_ADDLINK))
        {
            setEnable(false);
        }
        else if (roles.contains(ROLE_REMOVELINK))
        {
            setEnable(true);
        }
    }
}

IGraphsModel* ZDictPanel::graphsModel() const
{
    return m_model;
}

QModelIndex ZDictPanel::dictlistSocket() const
{
    return m_viewSockIdx;
}

ZenoSocketItem* ZDictPanel::socketItemByIdx(const QModelIndex& sockIdx) const
{
    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            if (pDictItem && pDictItem->socketIdx() == sockIdx)
            {
                return pDictItem->socketItem();
            }
        }
    }
    return nullptr;
}
