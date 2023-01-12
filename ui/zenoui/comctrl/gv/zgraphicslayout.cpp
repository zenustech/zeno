#include "zgraphicslayout.h"
#include <set>
#include <zassert.h>
#include "zenogvhelper.h"
#include "variantptr.h"
#include "zenoparamwidget.h"
#include "zveceditoritem.h"
#include "zdictpanel.h"


#define CURRENT_DEBUG_LAYOUT "dict"


ZGraphicsLayout::ZGraphicsLayout(bool bHor)
    : m_spacing(0)
    , m_parent(nullptr)
    , m_bHorizontal(bHor)
    , m_parentItem(nullptr)
{
}

void ZGraphicsLayout::setHorizontal(bool bHor)
{
    m_bHorizontal = bHor;
}

ZGraphicsLayout::~ZGraphicsLayout()
{
    clear();
}

void ZGraphicsLayout::addItem(QGraphicsItem* pItem)
{
    if (!pItem) return;

#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Item;
    item->pItem = pItem;
    item->pLayout = nullptr;
    item->pItem->setData(GVKEY_PARENT_LAYOUT, QVariant::fromValue((void*)this));
    pItem->setParentItem(m_parentItem);
    m_items.append(item);
}

void ZGraphicsLayout::addItem(QGraphicsItem* pItem, Qt::Alignment flag)
{
    if (!pItem) return;

#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Item;
    item->pItem = pItem;
    item->pLayout = nullptr;
    item->pItem->setData(GVKEY_PARENT_LAYOUT, QVariant::fromValue((void*)this));
    item->alignment = flag;
    pItem->setParentItem(m_parentItem);
    m_items.append(item);
}

void ZGraphicsLayout::insertItem(int i, QGraphicsItem* pItem)
{
    if (!pItem) return;

#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Item;
    item->pItem = pItem;
    item->pLayout = nullptr;
    item->pItem->setData(GVKEY_PARENT_LAYOUT, QVariant::fromValue((void*)this));
    pItem->setParentItem(m_parentItem);

    m_items.insert(i, item);
}

void ZGraphicsLayout::addLayout(ZGraphicsLayout* pLayout)
{
    if (!pLayout) return;

#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Layout;
    item->pItem = nullptr;
    item->pLayout = pLayout;
    pLayout->m_parent = this;
    pLayout->setParentItem(m_parentItem);
    m_items.append(item);
}

void ZGraphicsLayout::insertLayout(int i, ZGraphicsLayout* pLayout)
{
    if (!pLayout) return;

#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Layout;
    item->pItem = nullptr;
    item->pLayout = pLayout;
    pLayout->m_parent = this;
    pLayout->setParentItem(m_parentItem);

    m_items.insert(i, item);
}

ZGvLayoutItem* ZGraphicsLayout::itemAt(int idx) const
{
    if (idx < 0 || idx > m_items.size())
        return nullptr;
    return m_items[idx];
}

int ZGraphicsLayout::count() const
{
    return m_items.size();
}

void ZGraphicsLayout::setDebugName(const QString& dbgName)
{
    m_dbgName = dbgName;
}

void ZGraphicsLayout::addSpacing(qreal size)
{
#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Spacing;
    item->pItem = nullptr;
    item->pLayout = nullptr;
    if (size == -1)
    {
        item->gvItemSz.policy = m_bHorizontal ? QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed) :
                                    QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    }
    else
        item->gvItemSz.policy = QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    item->gvItemSz.minSize = m_bHorizontal ? QSizeF(size, 0) : QSizeF(0, size);

    m_items.append(item);
}

void ZGraphicsLayout::addSpacing(qreal sizehint, QSizePolicy policy)
{
#ifndef _DEBUG_ZLAYOUT
    QSharedPointer<ZGvLayoutItem> item(new ZGvLayoutItem);
#else
    ZGvLayoutItem* item = new ZGvLayoutItem;
#endif
    item->type = Type_Spacing;
    item->pItem = nullptr;
    item->pLayout = nullptr;
    if (m_bHorizontal)
    {
        if (policy.horizontalPolicy() == QSizePolicy::Minimum)
        {
            item->gvItemSz.policy = policy;
            item->gvItemSz.minSize = QSizeF(sizehint, 0);
        }
        else if (policy.horizontalPolicy() == QSizePolicy::Fixed)
        {
            item->gvItemSz.policy = policy;
            item->gvItemSz.minSize = QSizeF(sizehint, 0);
        }
        else if (policy.horizontalPolicy() == QSizePolicy::Expanding)
        {
            item->gvItemSz.policy = policy;
            item->gvItemSz.minSize = QSizeF(0, 0);
        }
    }
    else {
        //todo
    }
    m_items.append(item);
}


QGraphicsItem* ZGraphicsLayout::parentItem() const
{
    return m_parentItem;
}

ZGraphicsLayout* ZGraphicsLayout::parentLayout() const
{
    return m_parent;
}

void ZGraphicsLayout::setParentItem(QGraphicsItem* parent)
{
    m_parentItem = parent;
    //make every item in layout as the children of m_parentItem.
    for (auto item : m_items)
    {
        if (item->type == Type_Item) {
            item->pItem->setParentItem(parent);
        }
        else if (item->type == Type_Layout) {
            item->pLayout->setParentItem(parent);
        }
    }
}

ZGraphicsLayout* ZGraphicsLayout::parent() const
{
    return m_parent;
}

void ZGraphicsLayout::setSpacing(qreal spacing)
{
    m_spacing = spacing;
}

qreal ZGraphicsLayout::spacing() const
{
    return m_spacing;
}

void ZGraphicsLayout::setStretch(QList<int> stretchs)
{

}

void ZGraphicsLayout::setAlignment(QGraphicsItem* item, Qt::Alignment flag)
{
    for (auto _item : m_items)
    {
        if (_item->pItem == item) {
            _item->alignment = flag;
        }
    }
}

void ZGraphicsLayout::setContentsMargin(qreal top, qreal left, qreal bottom, qreal right)
{
    m_margins.setLeft(left);
    m_margins.setTop(top);
    m_margins.setRight(right);
    m_margins.setBottom(bottom);
}

QMargins ZGraphicsLayout::getContentsMargin() const
{
    return m_margins;
}

void ZGraphicsLayout::removeItem(QGraphicsItem* item)
{
    for (int i = 0; i < m_items.size(); i++)
    {
        if (m_items[i]->type == Type_Item && m_items[i]->pItem == item) {
            delete item;
            m_items.remove(i);
            break;
        }
    }
}

void ZGraphicsLayout::moveUp(int i)
{
    if (i < 1 || i > m_items.size()) {
        return;
    }

    auto tmp = m_items[i - 1];
    m_items[i - 1] = m_items[i];
    m_items[i] = tmp;
}

void ZGraphicsLayout::moveItem(int start, int destRow)
{
    if (start < 0 || destRow < 0 || start >= m_items.size() || destRow >= m_items.size() || start == destRow)
        return;

    if (start < destRow)
    {
        m_items.insert(destRow, m_items[start]);
        m_items.removeAt(start);
    }
    else
    {
        m_items.insert(destRow, m_items[start]);
        m_items.removeAt(start + 1);
    }
}

void ZGraphicsLayout::removeElement(int i)
{
    if (m_items[i]->type == Type_Layout) {
        removeLayout(m_items[i]->pLayout);
    } else {
        removeItem(m_items[i]->pItem);
    }
}

void ZGraphicsLayout::removeLayout(ZGraphicsLayout* layout)
{
    //delete item from layout.
    layout->clear();
    for (int i = 0; i < m_items.size(); i++)
    {
        if (m_items[i]->pLayout == layout) {
            m_items.remove(i);
            break;
        }
    }
}

void ZGraphicsLayout::clear()
{
    while (!m_items.isEmpty())
    {
        auto item = m_items.first();
        if (item->type == Type_Layout) {
            item->pLayout->clear();
        }
        else if (item->type == Type_Item) {
            delete item->pItem;
        }
        m_items.removeFirst();
    }
}

QSizeF ZGraphicsLayout::calculateSize()
{
    //todo: support fixed size directly for whole layout.
    QSizeF size(0, 0);

    QSizeF szMargin(m_margins.left() + m_margins.right(), m_margins.top() + m_margins.bottom());
    size += szMargin;

    if (m_dbgName == CURRENT_DEBUG_LAYOUT)
    {
        int j;
        j = 0;
    }

    for (int i = 0; i < m_items.size(); i++)
    {
        auto item = m_items[i];
        item->bDirty = true;
        switch (item->type)
        {
            case Type_Item:
            {
                if (item->pItem && !item->pItem->isVisible())
                    continue;

                //item with a layout, setup it recursively.
                ZGraphicsLayout* pLayout = QVariantPtr<ZGraphicsLayout>::asPtr(item->pItem->data(GVKEY_OWNLAYOUT));
                if (pLayout)
                {
                    QSizeF _size = pLayout->calculateSize();
                    if (m_bHorizontal) {
                        size.setHeight(qMax(_size.height() + szMargin.height(), size.height()));
                        size += QSizeF(_size.width(), 0);
                    }
                    else {
                        size.setWidth(qMax(_size.width() + szMargin.width(), size.width()));
                        size += QSizeF(0, _size.height());
                    }
                }
                else
                {
                    //use to debug
                    if (QGraphicsProxyWidget* pWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(item->pItem))
                    {
                        ZVecEditorItem* pVecEdit = qobject_cast<ZVecEditorItem*>(pWidget);
                        if (pVecEdit)
                        {
                            int j;
                            j = 0;
                        }
                    }

                    QSizeF sizeHint = ZenoGvHelper::sizehintByPolicy(item->pItem);
                    if (m_bHorizontal) {
                        size.setHeight(qMax(sizeHint.height() + szMargin.height(), size.height()));
                        size += QSizeF(sizeHint.width(), 0);
                    }
                    else {
                        size.setWidth(qMax(sizeHint.width() + szMargin.width(), size.width()));
                        size += QSizeF(0, sizeHint.height());
                    }
                }
                break;
            }
            case Type_Spacing:
            {
                if (m_bHorizontal) {
                    size.setHeight(qMax(item->gvItemSz.minSize.height() + szMargin.height(), size.height()));
                    size += QSizeF(item->gvItemSz.minSize.width(), 0);
                }
                else {
                    size.setWidth(qMax(item->gvItemSz.minSize.width() + szMargin.width(), size.width()));
                    size += QSizeF(0, item->gvItemSz.minSize.height());
                }
                break;
            }
            case Type_Layout:
            {
                QSizeF _size = item->pLayout->calculateSize();
                if (m_bHorizontal) {
                    size.setHeight(qMax(_size.height() + szMargin.height(), size.height()));
                    size += QSizeF(_size.width(), 0);
                }
                else {
                    size.setWidth(qMax(_size.width() + szMargin.width(), size.width()));
                    size += QSizeF(0, _size.height());
                }
                break;
            }
        }
        if (i < m_items.size() - 1)
        {
            if (m_bHorizontal)
            {
                size += QSizeF(m_spacing, 0);
            }
            else
            {
                size += QSizeF(0, m_spacing);
            }
        }
    }
    return size;
}

void ZGraphicsLayout::updateHierarchy(ZGraphicsLayout* pLayout)
{
    ZGraphicsLayout* rootLayout = visitRoot(pLayout);
    QSizeF sz = rootLayout->calculateSize();
    rootLayout->calcItemsSize(sz);

    QPointF pos(0, 0);  //by parent item.
    QRectF rc(0, 0, sz.width(), sz.height());

    rootLayout->setup(rc);

    //setup parent item.
    QGraphicsItem* pItem = rootLayout->parentItem();
    if (pItem)
    {
        SizeInfo info;
        info.pos = pItem->pos();
        info.minSize = sz;
        ZenoGvHelper::setSizeInfo(pItem, info);
        //z value manage.
        pItem->setZValue(-2);
    }
}

void ZGraphicsLayout::updateHierarchy(QGraphicsItem* pItem)
{
    //find the parent layout of this item.
    if (!pItem) return;

    if (ZGraphicsLayout* pLayout = QVariantPtr<ZGraphicsLayout>::asPtr(pItem->data(GVKEY_PARENT_LAYOUT)))
    {
        ZGraphicsLayout::updateHierarchy(pLayout);
    }
    else if (ZGraphicsLayout* pOwnLayout = QVariantPtr<ZGraphicsLayout>::asPtr(pItem->data(GVKEY_OWNLAYOUT)))
    {
        ZGraphicsLayout::updateHierarchy(pOwnLayout);
    }
    else
    {
        zeno::log_warn("the layout cannot be update hierarchically");
    }
}

ZGraphicsLayout* ZGraphicsLayout::visitRoot(ZGraphicsLayout* currentLayout)
{
    ZGraphicsLayout* root = currentLayout, * tmp = root;
    while (tmp)
    {
        while (tmp->parentLayout())
            tmp = tmp->parentLayout();

        root = tmp;
        auto item = tmp->parentItem();
        tmp = (ZGraphicsLayout*)item->data(GVKEY_PARENT_LAYOUT).value<void*>();
    }
    return root;
}

void ZGraphicsLayout::clearCacheSize()
{
    for (auto item : m_items)
    {
        if (item->type == Type_Layout) {
            item->pLayout->clearCacheSize();
        }
        item->bDirty = true;
    }
}

QRectF ZGraphicsLayout::geometry() const
{
    return m_geometry;
}

void ZGraphicsLayout::calcItemsSize(QSizeF layoutSize)
{
    auto getSize = [this](const QSizeF& sz) {
        return m_bHorizontal ? sz.width() : sz.height();
    };
    auto setSize = [this](QSizeF& sz, qreal val) {
        if (m_bHorizontal)
            sz.setWidth(val);
        else
            sz.setHeight(val);
    };

    QVector<int> szs(m_items.size(), 0);
    std::set<int> fixedItems, expandingItems;
    layoutSize = QSizeF(layoutSize.width() - m_margins.left() - m_margins.right(), layoutSize.height() - m_margins.top() - m_margins.bottom());

    qreal remaining = getSize(layoutSize);

    QSizeF szMargin(m_margins.left() + m_margins.right(), m_margins.top() + m_margins.bottom());
    remaining -= getSize(szMargin);

    for (int i = 0; i < m_items.size(); i++)
    {
        auto item = m_items[i];
        //clear cached size.
        item->actualSz = item->gvItemSz;

        ZGraphicsLayout* pLayout = nullptr;
        if (Type_Layout == item->type)
        {
            pLayout = item->pLayout;
            ZASSERT_EXIT(pLayout);
        }
        else if (Type_Item == item->type)
        {
            ZASSERT_EXIT(item->pItem);
            pLayout = QVariantPtr<ZGraphicsLayout>::asPtr(item->pItem->data(GVKEY_OWNLAYOUT));
            if (!item->pItem->isVisible())
                continue;
        }

        //use to debug.
        if (pLayout && pLayout->m_dbgName == CURRENT_DEBUG_LAYOUT)
        {
            int j;
            j = 0;
        }

        if (pLayout)
        {
            qreal sz = 0;
            if (item->bDirty) {
                QSizeF _layoutSize = pLayout->calculateSize();
                if (m_bHorizontal) {
                    pLayout->calcItemsSize(QSizeF(_layoutSize.width(), layoutSize.height()));
                }
                else {
                    pLayout->calcItemsSize(QSizeF(layoutSize.width(), _layoutSize.height()));
                }
                sz = getSize(_layoutSize);
                if (m_bHorizontal)
                {
                    item->actualSz.minSize.setWidth(sz);
                    item->actualSz.minSize.setHeight(layoutSize.height());
                }
                else
                {
                    item->actualSz.minSize.setWidth(layoutSize.width());
                    item->actualSz.minSize.setHeight(sz);
                }
                item->bDirty = false;
            }
            else {
                sz = getSize(item->actualSz.minSize);
            }
            remaining -= sz;
            fixedItems.insert(i);
            szs[i] = sz;
        }
        else
        {
            QGraphicsItem* pItem = item->pItem;

            if (QGraphicsProxyWidget* pWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(pItem))
            {
                ZenoVecEditItem* pVecEdit = qobject_cast<ZenoVecEditItem*>(pWidget);
                if (pVecEdit)
                {
                    int j;
                    j = 0;
                }
            }

            QSizePolicy policy = pItem ? pItem->data(GVKEY_SIZEPOLICY).value<QSizePolicy>() : item->actualSz.policy;
            QSizeF sizeHint = pItem ? ZenoGvHelper::sizehintByPolicy(pItem) : item->actualSz.minSize;

            item->actualSz.minSize = sizeHint;

            QSizePolicy::Policy hPolicy = policy.horizontalPolicy();
            QSizePolicy::Policy vPolicy = policy.verticalPolicy();

            if (m_bHorizontal) {
                if (vPolicy == QSizePolicy::Expanding) {
                    item->actualSz.minSize.setHeight(layoutSize.height());
                }
            }
            else {
                if (hPolicy == QSizePolicy::Expanding) {
                    item->actualSz.minSize.setWidth(layoutSize.width());    //minus margins.
                }
            }

            if ((m_bHorizontal && (hPolicy == QSizePolicy::Expanding || hPolicy == QSizePolicy::Minimum)) ||
                (!m_bHorizontal && (vPolicy == QSizePolicy::Expanding || vPolicy == QSizePolicy::Minimum)))
            {
                expandingItems.insert(i);
            }
            else
            {
                fixedItems.insert(i);
                qreal sz = getSize(sizeHint);
                remaining -= sz;
                setSize(item->actualSz.minSize, sz);
                item->bDirty = false;
                szs[i] = sz;
            }
        }
        if (i < m_items.size() - 1)
            remaining -= m_spacing;
    }
    //remaining space allocated for all expanding widget.
    if (!expandingItems.empty())
    {
        int n = expandingItems.size();
        qreal sz = remaining / n;
        for (int i : expandingItems)
        {
            szs[i] = sz;
            setSize(m_items[i]->actualSz.minSize, sz);
            m_items[i]->bDirty = false;
        }
    }
}

void ZGraphicsLayout::setup(QRectF rc)
{
    if (m_dbgName == CURRENT_DEBUG_LAYOUT)
    {
        int j;
        j = 0;
    }

    //set geometry relative to item which owns this layout, indicated by rc.
    m_geometry = rc;
    rc = rc.marginsRemoved(m_margins);
    qreal xPos = rc.topLeft().x(), yPos = rc.topLeft().y();
    //set geometry for each component.
    for (int i = 0; i < m_items.size(); i++)
    {
        auto item = m_items[i];
        switch (item->type)
        {
            case Type_Layout:
            case Type_Item:
            case Type_Spacing:
            {
                QSizeF sz = item->actualSz.minSize;
                QRectF _rc(QPointF(xPos, yPos), sz);

                if (item->type == Type_Layout)
                {
                    ZASSERT_EXIT(item->pLayout);
                    item->pLayout->setup(_rc);
                }
                else if (item->type == Type_Item)
                {
                    SizeInfo info;
                    info.minSize = sz;
                    info.pos.setX(xPos);
                    info.pos.setY(yPos);

                    if (item->alignment & Qt::AlignRight)
                    {
                        info.pos.setX(rc.right() - sz.width());
                    }
                    else if (item->alignment & Qt::AlignHCenter)
                    {
                        info.pos.setX(rc.center().x() - sz.width() / 2);
                    }

                    if (item->alignment & Qt::AlignVCenter)
                    {
                        info.pos.setY(rc.center().y() - sz.height() / 2);
                    }

                    ZenoGvHelper::setSizeInfo(item->pItem, info);

                    //item with a layout, setup it recursively.
                    ZGraphicsLayout* pLayout = QVariantPtr<ZGraphicsLayout>::asPtr(item->pItem->data(GVKEY_OWNLAYOUT));
                    if (pLayout) {
                        QRectF __rc(0, 0, _rc.width(), _rc.height());
                        pLayout->setup(__rc);
                    }
                }

                if (m_bHorizontal)
                    xPos += sz.width();
                else
                    yPos += sz.height();
                break;
            }
        }
        if (m_bHorizontal)
            xPos += m_spacing;
        else
            yPos += m_spacing;
    }
}