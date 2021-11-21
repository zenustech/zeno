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
class QDMNodeParamEdit;

class QDMGraphicsNode : public QGraphicsItem
{
    friend Interceptor;

    std::vector<std::unique_ptr<QDMGraphicsSocketIn>> socketIns;
    std::vector<std::unique_ptr<QDMGraphicsSocketOut>> socketOuts;

    std::unique_ptr<QGraphicsTextItem> label;
    dop::Descriptor const *desc{};
    std::string name;

    [[nodiscard]] float getHeight() const;

protected:
    QDMGraphicsSocketIn *addSocketIn();
    QDMGraphicsSocketOut *addSocketOut();
    void initByDescriptor(const dop::Descriptor &desc);

public:
    QDMGraphicsNode();
    void initByType(std::string const &type);
    //void socketUnlinked(QDMGraphicsSocketIn *socket);
    //void socketLinked(QDMGraphicsSocketIn *socket, QDMGraphicsSocketOut *srcSocket);
    virtual void setupParamEdit(QDMNodeParamEdit *paredit);

    ~QDMGraphicsNode() override;
    [[nodiscard]] QRectF boundingRect() const override;
    void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

    [[nodiscard]] QDMGraphicsSocketIn *socketInAt(size_t index);
    [[nodiscard]] QDMGraphicsSocketOut *socketOutAt(size_t index);
    size_t socketInIndex(QDMGraphicsSocketIn *socket);
    size_t socketOutIndex(QDMGraphicsSocketOut *socket);
    [[nodiscard]] std::vector<std::string> getInputNames() const;
    [[nodiscard]] std::vector<std::string> getOutputNames() const;

    //void socketValueChanged(QDMGraphicsSocketIn *socket);
    [[nodiscard]] virtual QDMGraphicsScene *getSubnetScene() const;
    [[nodiscard]] QDMGraphicsScene *getScene() const;

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
