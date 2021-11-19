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
#include <zeno/dop/Descriptor.h>
#include <QWidget>
#include <memory>

ZENO_NAMESPACE_BEGIN

struct Interceptor;
class QDMGraphicsScene;

class QDMGraphicsNode : public QGraphicsItem
{
    friend Interceptor;

    std::vector<std::unique_ptr<QDMGraphicsSocketIn>> socketIns;
    std::vector<std::unique_ptr<QDMGraphicsSocketOut>> socketOuts;
    std::unique_ptr<QDMGraphicsScene> subnetScene;
    std::unique_ptr<dop::Descriptor> subnetDesc{};

    std::unique_ptr<QGraphicsTextItem> label;
    dop::Descriptor const *desc{};
    std::string name;

    void initByDescriptor(dop::Descriptor const &desc);
    float getHeight() const;

public:
    QDMGraphicsNode();
    ~QDMGraphicsNode();

    virtual QRectF boundingRect() const override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

    QDMGraphicsSocketIn *socketInAt(size_t index);
    QDMGraphicsSocketOut *socketOutAt(size_t index);
    size_t socketInIndex(QDMGraphicsSocketIn *socket);
    size_t socketOutIndex(QDMGraphicsSocketOut *socket);

    //void socketUnlinked(QDMGraphicsSocketIn *socket);
    //void socketLinked(QDMGraphicsSocketIn *socket, QDMGraphicsSocketOut *srcSocket);
    //void socketValueChanged(QDMGraphicsSocketIn *socket);

    QDMGraphicsSocketIn *addSocketIn();
    QDMGraphicsSocketOut *addSocketOut();

    void initAsSubnet();
    void initByType(QString type);

    QDMGraphicsScene *getSubnetScene() const;
    QDMGraphicsScene *getScene() const;
    void setName(QString name);

    inline std::string const &getName() { return name; }
    inline dop::Descriptor const *getDescriptor() { return desc; }

    void invalidate();
    void unlinkAll();

    static constexpr float WIDTH = 200, HEIGHT = 60, ROUND = 6, BORDER = 3;
    static constexpr float SOCKMARGINTOP = 20, SOCKSTRIDE = 30, SOCKMARGINBOT = -10;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSNODE_H
