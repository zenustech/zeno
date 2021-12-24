#ifndef __ZENO_BACKGROUND_ITEM_H__
#define __ZENO_BACKGROUND_ITEM_H__

#include <QtWidgets>
#include "zenosvgitem.h"

class ZenoBackgroundItem : public QGraphicsObject
{
    typedef QGraphicsObject _base;
    Q_OBJECT
public:
    ZenoBackgroundItem(const BackgroundComponent &comp, QGraphicsItem *parent = nullptr);
    QRectF boundingRect() const override;
    void resize(QSizeF sz);
    void setColors(const QColor &clrNormal, const QColor &clrHovered, const QColor &clrSelected);
    void setRadius(int lt, int rt, int lb, int rb);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    void toggle(bool bSelected);

private:
    QPainterPath shape() const override;

    int lt_radius, rt_radius, lb_radius, rb_radius;
    QColor m_clrNormal, m_clrHovered, m_clrSelected;
    QRectF m_rect;
    ZenoImageItem *m_img;
    bool m_bFixRadius;
    bool m_bSelected;
};

class ZenoBackgroundWidget : public QGraphicsWidget
{
    typedef QGraphicsWidget _base;
    Q_OBJECT
public:
    ZenoBackgroundWidget(QGraphicsItem *parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    QRectF boundingRect() const override;
    void setGeometry(const QRectF &rect) override;
    void setColors(const QColor &clrNormal, const QColor &clrHovered, const QColor &clrSelected);
    void setRadius(int lt, int rt, int lb, int rb);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    void toggle(bool bSelected);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
    QPainterPath shape() const override;

    int lt_radius, rt_radius, lb_radius, rb_radius;
    QColor m_clrNormal, m_clrHovered, m_clrSelected;
    QColor m_color;
    bool m_bFixRadius;
    bool m_bSelected;
};

#endif