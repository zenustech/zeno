#ifndef QDMGRAPHICSNODE_H
#define QDMGRAPHICSNODE_H

#include <QGraphicsItem>
#include <vector>
#include "qdmgraphicssocketin.h"
#include "qdmgraphicssocketout.h"
#include <QRectF>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsTextItem>
#include <QWidget>
#include <zeno/dop/Node.h>
#include <memory>

ZENO_NAMESPACE_BEGIN

class QDMGraphicsNode : public QGraphicsItem
{
    std::vector<std::unique_ptr<QDMGraphicsSocketIn>> socketIns;
    std::vector<std::unique_ptr<QDMGraphicsSocketOut>> socketOuts;
    std::unique_ptr<QGraphicsTextItem> label;

    std::unique_ptr<dop::Node> dopNode;

public:
    QDMGraphicsNode();
    ~QDMGraphicsNode();

    float getHeight() const;
    dop::Node *getDopNode() const;

    virtual QRectF boundingRect() const override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

    size_t socketInIndex(QDMGraphicsSocketIn *socket);
    size_t socketOutIndex(QDMGraphicsSocketOut *socket);
    void socketUnlinked(QDMGraphicsSocketIn *socket);
    void socketLinked(QDMGraphicsSocketIn *socket, QDMGraphicsSocketOut *srcSocket);
    void unlinkAll();

    QDMGraphicsSocketIn *addSocketIn();
    QDMGraphicsSocketOut *addSocketOut();
    void initByName(QString name);
    void setName(QString name);

    static constexpr float WIDTH = 200, HEIGHT = 60, ROUND = 6, BORDER = 3;
    static constexpr float SOCKMARGINTOP = 20, SOCKSTRIDE = 30, SOCKMARGINBOT = -10;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSNODE_H
