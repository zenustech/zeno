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
    ResizableItemImpl(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    QRectF boundingRect() const override;
    QRectF coreItemSceneRect();
    int width() const { return m_width; }
    int height() const { return m_height; }
    void setCoreItem(ResizableCoreItem* pItem);
    void showBorder(bool bShow);

protected:
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

signals:
    void itemGeoChanged(QRectF sceneRect);

private:
    void _adjustItemsPos();
    bool _enableMouseEvent();

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

    qreal m_width;
    qreal m_height;

    bool m_showBdr;
};


#endif