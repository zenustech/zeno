#ifndef __ZGRAPHICS_LAYOUT_WIDGETS_H__
#define __ZGRAPHICS_LAYOUT_WIDGETS_H__

#include "zgraphicslayoutitem.h"
#include <QGraphicsWidget>

class ZLayoutBackground : public ZGraphicsLayoutItem<QGraphicsWidget>
{
    typedef ZGraphicsLayoutItem<QGraphicsWidget> _base;
    Q_OBJECT
public:
    ZLayoutBackground(QGraphicsItem* parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    ~ZLayoutBackground();
    QRectF boundingRect() const override;
    void setBorder(qreal width, const QColor& clrBorder);
    void setGeometry(const QRectF& rect) override;
    void setColors(bool bAcceptHovers, const QColor& clrNormal, const QColor& clrHovered = QColor(), const QColor& clrSelected = QColor());
    void setRadius(int lt, int rt, int lb, int rb);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    void toggle(bool bSelected);

signals:
    void doubleClicked();
    void hoverEntered();
    void hoverLeaved();

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF& constraint = QSizeF()) const override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;

private:
    QPainterPath shape() const override;

    qreal m_borderWidth;
    int lt_radius, rt_radius, lb_radius, rb_radius;

    QColor m_clrNormal, m_clrHovered, m_clrSelected;
    QColor m_color;
    QColor m_clrBorder;
    bool m_bFixRadius;
    bool m_bSelected;
};



#endif