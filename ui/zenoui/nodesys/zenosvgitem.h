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
    QSizeF size() const { return m_size; }
    bool isHovered() const;
    void setCheckable(bool bCheckable);

signals:
    void clicked();
    void toggled(bool);
    void hoverChanged(bool);

public slots:
    void setHovered(bool bHovered);
    void toggle(bool bSelected);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    QString m_normal;
    QString m_hovered;
    QString m_selected;
    ZenoSvgItem* m_svg;
    QSizeF m_size;
    bool m_bToggled;
    bool m_bHovered;
    bool m_bCheckable;
};

#endif