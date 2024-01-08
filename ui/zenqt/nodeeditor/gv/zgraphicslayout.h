#ifndef __ZGRAPHICSLAYOUT_H__
#define __ZGRAPHICSLAYOUT_H__

#include <QtWidgets>
#include <QSizePolicy>

#define _DEBUG_ZLAYOUT

class ZGraphicsLayout;
class QGraphicsItem;

enum LayoutItemType
{
    Type_Unknown,
    Type_Item,
    Type_Layout,
    Type_Spacing,
    Type_Stretch,
};

enum CUSTOM_DATA_KEY
{
    GVKEY_SIZEHINT,
    GVKEY_BOUNDING,     //for bounding rect size.
    GVKEY_SIZEPOLICY,
    GVKEY_OWNLAYOUT,
    GVKEY_PARENT_LAYOUT,
    GVKEY_TYPE,         //socket/param type.
    GVKEY_CONTROL,      //control widget type, see PARAM_CONTROL.
    GVKEY_ALIGNMENT,
};

struct SizeInfo
{
    QPointF pos;
    QSizeF minSize;
    QSizeF maxSize;
    QSizeF preferSize;
    QSizePolicy policy;
};

struct ZGvLayoutItem
{
    SizeInfo gvItemSz;      //user specific size info.
    SizeInfo actualSz;      //layout calculated cached info.
    ZGraphicsLayout* pLayout;
    QGraphicsItem* pItem;
    Qt::Alignment alignment;
    LayoutItemType type;
    bool bDirty;

    ZGvLayoutItem()
        : pLayout(nullptr)
        , pItem(nullptr)
        , type(Type_Unknown)
        , bDirty(true)
    {}
};

class ZGraphicsLayout
{
public:
    ZGraphicsLayout(bool bHor);
    ~ZGraphicsLayout();
    void addItem(QGraphicsItem* item);
    void addItem(QGraphicsItem* item, Qt::Alignment flag);
    int count() const;
    ZGvLayoutItem* itemAt(int idx) const;
    void addLayout(ZGraphicsLayout* pLayout);
    void insertLayout(int i, ZGraphicsLayout* pLayout);
    void insertItem(int i, QGraphicsItem* pItem);
    void addSpacing(qreal size);
    void addSpacing(qreal sizehint, QSizePolicy policy);
    void setParentItem(QGraphicsItem* item);
    QGraphicsItem* parentItem() const;
    ZGraphicsLayout* parentLayout() const;
    void setStretch(QList<int> stretchs);
    void setAlignment(QGraphicsItem* item, Qt::Alignment flag);
    void setSpacing(qreal spacing);
    void setHorizontal(bool bHor);
    qreal spacing() const;
    void setContentsMargin(qreal top, qreal left, qreal bottom, qreal right);
    QMargins getContentsMargin() const;
    QRectF geometry() const;
    void moveUp(int i);
    void moveItem(int i, int j);
    void removeElement(int i);
    void removeItem(QGraphicsItem* item);
    void removeLayout(ZGraphicsLayout* layout);
    void clear();
    ZGraphicsLayout* parent() const;
    static void updateHierarchy(ZGraphicsLayout* pLayout);
    static void updateHierarchy(QGraphicsItem* pItem);
    static ZGraphicsLayout* visitRoot(ZGraphicsLayout* currentLayout);
    void clearCacheSize();
    void setDebugName(const QString& dbgName);

private:
    void setup(QRectF rc);
    QSizeF calculateSize();
    void calcItemsSize(QSizeF layoutSize);

#ifdef _DEBUG_ZLAYOUT
    QVector<ZGvLayoutItem*> m_items;
    QString m_dbgName;
#else
    QVector<QSharedPointer<ZGvLayoutItem>> m_items;
#endif
    ZGraphicsLayout* m_parent;
    QGraphicsItem* m_parentItem;
    QRectF m_geometry;
    qreal m_spacing;
    QMargins m_margins;
    bool m_bHorizontal;
};

#endif