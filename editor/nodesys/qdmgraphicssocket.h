#ifndef QDMGRAPHICSSOCKET_H
#define QDMGRAPHICSSOCKET_H

#include <zeno/common.h>
#include <QGraphicsItem>
#include <QGraphicsTextItem>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QRectF>
#include <set>

ZENO_NAMESPACE_BEGIN

class QDMGraphicsLinkFull;

class QDMGraphicsSocket : public QGraphicsItem
{
    std::set<QDMGraphicsLinkFull *> links;
    std::string name;
    std::string type;
    std::string defl;

protected:
    QGraphicsTextItem *label;

public:
    QDMGraphicsSocket();

    virtual void unlinkAll();
    virtual void linkRemoved(QDMGraphicsLinkFull *link);
    virtual void linkAttached(QDMGraphicsLinkFull *link);
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
    virtual QRectF boundingRect() const override;
    virtual QPointF getLinkedPos() const = 0;
    inline std::string const &getName() const { return name; }
    inline std::string const &getType() const { return type; }
    inline std::string const &getDefl() const { return defl; }
    void setType(std::string const &type);
    void setName(std::string const &name);
    void setDefl(std::string const &defl);

    static constexpr float SIZE = 20, ROUND = 4;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSOCKET_H
