#ifndef QDMGRAPHICSLINK_H
#define QDMGRAPHICSLINK_H

#include <QGraphicsItem>
#include <QRectF>
#include <QPainter>
#include <QPainterPath>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QPointF>

class QDMGraphicsLink : public QGraphicsItem
{
    mutable QPointF lastSrcPos, lastDstPos;
    mutable bool hasLastPath{false};
    mutable QPainterPath lastPath;

public:
    QDMGraphicsLink();

    virtual QRectF boundingRect() const override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    virtual QPainterPath shape() const override;

    virtual QPointF getSrcPos() const = 0;
    virtual QPointF getDstPos() const = 0;

    static constexpr float BEZIER = 0.5f, WIDTH = 3;
};

#endif // QDMGRAPHICSLINK_H
