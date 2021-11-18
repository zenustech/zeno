#ifndef __RESIZABLE_RECT_ITEM_H__
#define __RESIZABLE_RECT_ITEM_H__

class ResizableRectItem : public QObject
                        , public QGraphicsRectItem
{
    Q_OBJECT
    typedef QGraphicsRectItem _base;

    enum MOUSE_HINT
    {
        MOUSE_DONOTHING,

        SCALE_LEFT_TOP,
        SCALE_LEFT_MID,
        SCALE_LEFT_BOTTOM,

        SCALE_MID_TOP,
        SCALE_MID_BOTTOM,

        SCALE_RIGHT_TOP,
        SCALE_RIGHT_MID,
        SCALE_RIGHT_BOTTOM,

        TRANSLATE,
        OUTSIDE,
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
    ResizableRectItem(qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;
    QRectF boundingRect() const override;

protected:
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    void _adjustItemsPos();

    const qreal dragW = 8.;
    const qreal dragH = 8.;
    const qreal borderW = 2.;

    QGraphicsRectItem* m_ltcorner;
    QGraphicsRectItem* m_rtcorner;
    QGraphicsRectItem* m_lbcorner;
    QGraphicsRectItem* m_rbcorner;

    MOUSE_HINT m_mouseHint;
    SCALE_INFO m_movescale_info;
};


#endif