#ifndef __RESIZABLE_ITEM_IMPL_H__
#define __RESIZABLE_ITEM_IMPL_H__

#include "nodescene.h"

class ResizableCoreItem;

class ResizableItemImpl : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;

    enum DRAG_ITEM
    {
        DRAG_LEFTTOP,
        DRAG_LEFTMID,
        DRAG_LEFTBOTTOM,
        DRAG_MIDTOP,
        DRAG_MIDBOTTOM,
        DRAG_RIGHTTOP,
        DRAG_RIGHTMID,
        DRAG_RIGHTBOTTOM,
        TRANSLATE,
        NO_DRAG,
    };

    struct SCALE_INFO {
        qreal old_width;
        qreal old_height;

        QPointF fixed_point;
        qreal fixed_x;	//left
        qreal fixed_y;	//top

        SCALE_INFO() : old_width(0), old_height(0), fixed_x(0), fixed_y(0) {}
    };

public:
    ResizableItemImpl(NODE_TYPE type, const QString& id, const QRectF& sceneRc, QGraphicsItem* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    QRectF boundingRect() const override;
    QRectF coreItemSceneRect();
    void setCoreItemSceneRect(const QRectF& rc);
    int width() const { return m_width; }
    int height() const { return m_height; }
    void setCoreItem(ResizableCoreItem* pItem);
    ResizableCoreItem *coreItem() const { return m_coreitem; }
    void showBorder(bool bShow);
    void setLocked(bool bLock);
    void setContent(NODE_CONTENT content) { m_content = content; }
    bool isLocked() const;
    QString getId() const;
    NODE_TYPE getType() const { return m_type; }
    NODE_CONTENT getContent() const { return m_content; }
    void resetZValue();

protected:
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

signals:
    void itemDeselected();
    void itemSelectedChange(NODE_ID id, bool bSelected);
    void gvItemGeoChanged(QString id, QRectF sceneRect);
    void gvItemSelectedChange(QString id, bool selected);

private:
    void _resetDragPoints();
    void _adjustItemsPos();
    bool _enableMouseEvent();
    void _sizeValidate(bool bTranslate);
    void _setPosition(QPointF pos);

    QGraphicsItem* getResizeHandleItem(QPointF scenePos);

    const qreal dragW = 6.;
    const qreal dragH = 6.;
    const qreal borderW = 1.;

    QGraphicsRectItem* m_borderitem;
    QVector<QGraphicsRectItem*> m_dragPoints;
    std::unordered_map<DRAG_ITEM, Qt::CursorShape> m_cursor_mapper;

    ResizableCoreItem* m_coreitem;

    QPixmap m_originalPix;

    DRAG_ITEM m_mouseHint;
    SCALE_INFO m_movescale_info;

    const QString m_id;

    qreal m_width;
    qreal m_height;

    NODE_TYPE m_type;
    NODE_CONTENT m_content;

    bool m_showBdr;
    bool m_bLocked;
};


#endif