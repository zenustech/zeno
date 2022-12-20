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


class ZDictItemLayout : public ZGraphicsLayout
{
public:
    ZDictItemLayout(const QModelIndex& keyIdx, const CallbackForSocket& cbSock)
        : ZGraphicsLayout(true)
        , m_idx(keyIdx)
        , m_editText(nullptr)
    {
        ImageElement elem;
        elem.image = ":/icons/socket-off.svg";
        elem.imageHovered = ":/icons/socket-hover.svg";
        elem.imageOn = ":/icons/socket-on.svg";
        elem.imageOnHovered = ":/icons/socket-on-hover.svg";

        const int cSocketWidth = ZenoStyle::dpiScaled(12);
        const int cSocketHeight = ZenoStyle::dpiScaled(12);

        m_socket = new ZenoSocketItem(m_idx, true, elem, QSizeF(cSocketWidth, cSocketHeight));
        qreal leftMargin = ZenoStyle::dpiScaled(10);
        qreal rightMargin = ZenoStyle::dpiScaled(10);
        qreal topMargin = ZenoStyle::dpiScaled(10);
        qreal bottomMargin = ZenoStyle::dpiScaled(10);
        m_socket->setContentMargins(leftMargin, topMargin, rightMargin, bottomMargin);

        QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() {
            if (cbSock.cbOnSockClicked)
                cbSock.cbOnSockClicked(m_socket);
        });

        //move up button
        elem.image = ":/icons/moveUp.svg";
        elem.imageHovered = ":/icons/moveUp-on.svg";
        elem.imageOn = ":/icons/moveUp.svg";
        elem.imageOnHovered = ":/icons/moveUp-on.svg";
        ZenoImageItem* pMoveUpBtn = new ZenoImageItem(elem, ZenoStyle::dpiScaledSize(QSizeF(20, 20)));
        QObject::connect(pMoveUpBtn, &ZenoImageItem::clicked, [=]() {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_idx.model());
            int r = m_idx.row();
            if (r > 0) {
                const QModelIndex& parent = m_idx.parent();
                pModel->moveRow(parent, r, parent, r - 1);
            }
        });

        //close button
        elem.image = ":/icons/closebtn.svg";
        elem.imageHovered = ":/icons/closebtn_on.svg";
        elem.imageOn = ":/icons/closebtn.svg";
        elem.imageOnHovered = ":/icons/closebtn_on.svg";
        ZenoImageItem* pRemoveBtn = new ZenoImageItem(elem, ZenoStyle::dpiScaledSize(QSizeF(20,20)));
        QObject::connect(pRemoveBtn, &ZenoImageItem::clicked, [=]() {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_idx.model());
            pModel->removeRow(m_idx.row());
        });

        Callback_EditFinished cbEditFinished = [=](QVariant newValue) {
            if (newValue == m_idx.data().toString())
                return;
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_idx.model());
            pModel->setData(m_idx, newValue, Qt::DisplayRole);
        };

        const QString& key = m_idx.data().toString();
        m_editText = zenoui::createItemWidget(key, CONTROL_STRING, "string", cbEditFinished, nullptr, CALLBACK_SWITCH(), QVariant());

        addItem(m_socket, Qt::AlignVCenter);
        addItem(m_editText, Qt::AlignVCenter);
        addItem(pMoveUpBtn, Qt::AlignVCenter);
        addItem(pRemoveBtn, Qt::AlignVCenter);
        setSpacing(ZenoStyle::dpiScaled(5));
    }
    ZenoSocketItem* socketItem() const
    {
        return m_socket;
    }
    QPersistentModelIndex socketIdx() const
    {
        return m_idx;
    }
    void updateName(const QString& newKeyName)
    {
        ZenoGvHelper::setValue(m_editText, CONTROL_STRING, newKeyName);
    }

private:
    QPersistentModelIndex m_idx;
    ZenoSocketItem* m_socket;
    QGraphicsItem *m_editText;
};


ZDictPanel::ZDictPanel(const QPersistentModelIndex& viewSockIdx, const CallbackForSocket& cbSock)
    : ZLayoutBackground()
    , m_viewSockIdx(viewSockIdx)
{
    int radius = ZenoStyle::dpiScaled(0);
    setRadius(radius, radius, radius, radius);
    setColors(false, QColor(50, 50, 50), QColor(50, 50, 50), QColor(50, 50, 50));
    setBorder(0, QColor());

    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);

    pVLayout->setContentsMargin(8, 0, 8, 8);
    pVLayout->setSpacing(ZenoStyle::dpiScaled(8));

    QAbstractItemModel* pKeyObjModel = QVariantPtr<QAbstractItemModel>::asPtr(m_viewSockIdx.data(ROLE_VPARAM_LINK_MODEL));
    for (int r = 0; r < pKeyObjModel->rowCount(); r++)
    {
        const QModelIndex& idxKey = pKeyObjModel->index(r, 0);
        QString key = idxKey.data().toString();

        ZDictItemLayout* pkey = new ZDictItemLayout(idxKey, cbSock);
        pVLayout->addLayout(pkey);
    }

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("+ Add Dict Key", "blueStyle");
    pVLayout->addItem(pEditBtn, Qt::AlignHCenter);

    QObject::connect(pEditBtn, &ZenoParamPushButton::clicked, [=]() {
        QStringList keys;
        int n = pKeyObjModel->rowCount();
        for (int r = 0; r < pKeyObjModel->rowCount(); r++)
        {
            const QModelIndex& idxKey = pKeyObjModel->index(r, 0);
            keys.append(idxKey.data().toString());
        }
        const QString &newKeyName = UiHelper::getUniqueName(keys, "obj", false);

        pKeyObjModel->insertRow(n);
        QModelIndex newIdx = pKeyObjModel->index(n, 0);
        pKeyObjModel->setData(newIdx, newKeyName, Qt::DisplayRole);
    });

    QObject::connect(pKeyObjModel, &QAbstractItemModel::rowsInserted, [=](const QModelIndex& parent, int start, int end) {
        const QModelIndex& idxKey = pKeyObjModel->index(start, 0);
        QString key = idxKey.data().toString();
        ZDictItemLayout* pkey = new ZDictItemLayout(idxKey, cbSock);
        pVLayout->insertLayout(start, pkey);

        ZGraphicsLayout::updateHierarchy(pVLayout);
        if (cbSock.cbOnSockLayoutChanged)
            cbSock.cbOnSockLayoutChanged();
    });

    QObject::connect(pKeyObjModel, &QAbstractItemModel::rowsAboutToBeRemoved,
                     [=](const QModelIndex &parent, int first, int last) {
        pVLayout->removeElement(first);
        ZGraphicsLayout::updateHierarchy(pVLayout);
        if (cbSock.cbOnSockLayoutChanged)
            cbSock.cbOnSockLayoutChanged();
    });

    QObject::connect(pKeyObjModel, &QAbstractItemModel::rowsMoved,
        [=](const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row) {
            //only support move up for now.
            pVLayout->moveUp(start);
            ZGraphicsLayout::updateHierarchy(pVLayout);
            if (cbSock.cbOnSockLayoutChanged)
                cbSock.cbOnSockLayoutChanged();
    });

    QObject::connect(pKeyObjModel, &QAbstractItemModel::dataChanged,
        [=](const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles) {
            if (roles.contains(Qt::DisplayRole) && topLeft.column() == 0) {
                ZDictItemLayout* pItemLayout = static_cast<ZDictItemLayout*>(pVLayout->itemAt(topLeft.row())->pLayout);
                const QString& newKeyName = topLeft.data(Qt::DisplayRole).toString();
                pItemLayout->updateName(newKeyName);
            }
    });

    setLayout(pVLayout);
}

ZenoSocketItem* ZDictPanel::socketItemByIdx(const QModelIndex& sockIdx) const
{
    for (int i = 0; i < m_layout->count(); i++)
    {
        auto layoutItem = m_layout->itemAt(i);
        if (layoutItem->type == Type_Layout)
        {
            ZDictItemLayout* pDictItem = static_cast<ZDictItemLayout*>(layoutItem->pLayout);
            if (pDictItem->socketIdx() == sockIdx)
            {
                return pDictItem->socketItem();
            }
        }
    }
    return nullptr;
}
