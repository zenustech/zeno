#ifndef QDMGRAPHICSSOCKET_H
#define QDMGRAPHICSSOCKET_H

#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QRectF>
#include <set>

class QDMGraphicsLinkFull;

class QDMGraphicsSocket : public QGraphicsItem
{
    std::set<QDMGraphicsLinkFull *> links;
    QGraphicsTextItem *label;

public:
    QDMGraphicsSocket();

    void unlinkAll();
    void linkRemoved(QDMGraphicsLinkFull *link);
    virtual void linkAttached(QDMGraphicsLinkFull *link);
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    virtual QRectF boundingRect() const override;
    void setName(QString name);

    static constexpr float SIZE = 20, ROUND = 4;
};

#endif // QDMGRAPHICSSOCKET_H
