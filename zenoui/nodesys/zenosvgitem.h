#ifndef __ZENO_SVG_ITEM_H__
#define __ZENO_SVG_ITEM_H__

#include <QtSvg/QGraphicsSvgItem>
#include "../render/renderparam.h"

class ZenoSvgItem : public QGraphicsSvgItem
{
public:
    ZenoSvgItem(QGraphicsItem *parent = 0);
    ZenoSvgItem(const QString &normal, QGraphicsItem *parent = 0);

    void setSize(QSizeF size);
    void setSize(qreal width, qreal height) {
        setSize(QSizeF(width, height));
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);
    QRectF boundingRect() const override;

private:
    QSizeF m_size;
};

class ZenoImageItem : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;
public:
    ZenoImageItem(const ImageElement &elem, const QSizeF& sz, QGraphicsItem *parent = 0);
    ZenoImageItem(const QString &normal, const QString &hovered, const QString &selected, const QSizeF &sz, QGraphicsItem *parent = 0);
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;
    void resize(QSizeF sz);
    void toggle(bool bSelected);

signals:
    void clicked();
    void toggled(bool);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

private:
    QString m_normal;
    QString m_hovered;
    QString m_selected;
    ZenoSvgItem* m_svg;
    QSizeF m_size;
};

#endif