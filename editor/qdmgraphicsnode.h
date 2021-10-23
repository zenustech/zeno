#ifndef QDMGRAPHICSNODE_H
#define QDMGRAPHICSNODE_H

#include <QGraphicsItem>
#include <vector>
#include <QPointer>
#include "qdmgraphicssocketin.h"
#include "qdmgraphicssocketout.h"
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QRectF>

class QDMGraphicsNode : public QGraphicsItem
{
    std::vector<QDMGraphicsSocketIn *> socketIns;
    std::vector<QDMGraphicsSocketOut *> socketOuts;

public:
    QDMGraphicsNode();
    ~QDMGraphicsNode();

    virtual QRectF boundingRect() const override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;

    QDMGraphicsSocketIn *addSocketIn();
    QDMGraphicsSocketOut *addSocketOut();

    static constexpr float WIDTH = 200, HEIGHT = 60, ROUND = 6, BORDER = 3;
    static constexpr float SOCKMARGINTOP = 25, SOCKSTRIDE = 30, SOCKMARGINBOT = -5;
};

#endif // QDMGRAPHICSNODE_H
