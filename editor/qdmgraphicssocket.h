#ifndef QDMGRAPHICSSOCKET_H
#define QDMGRAPHICSSOCKET_H

#include <QGraphicsItem>
#include <QGraphicsTextItem>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QRectF>
#include <set>

class QDMGraphicsLinkFull;

class QDMGraphicsSocket : public QGraphicsItem
{
    std::set<QDMGraphicsLinkFull *> links;

protected:
    QGraphicsTextItem *label;

public:
    QDMGraphicsSocket();

    void unlinkAll();
    void linkRemoved(QDMGraphicsLinkFull *link);
    virtual void linkAttached(QDMGraphicsLinkFull *link);
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    virtual QRectF boundingRect() const override;
    virtual QPointF getLinkedPos() const = 0;
    void setName(QString name);

    static constexpr float SIZE = 20, ROUND = 4;
};

#endif // QDMGRAPHICSSOCKET_H
