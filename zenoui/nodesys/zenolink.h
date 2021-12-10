#ifndef __ZENO_LINK_H__
#define __ZENO_LINK_H__

#include <QtWidgets>

class ZenoSubGraphScene;

class ZenoLink : public QGraphicsItem
{
    typedef QGraphicsItem _base;

public:
    ZenoLink(QGraphicsItem* parent = nullptr);

    virtual QRectF boundingRect() const override;
    virtual QPainterPath shape() const override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;

    virtual QPointF getSrcPos() const = 0;
    virtual QPointF getDstPos() const = 0;

private:
    static constexpr float BEZIER = 0.5f, WIDTH = 3;

    mutable QPointF lastSrcPos, lastDstPos;
    mutable bool hasLastPath{false};
    mutable QPainterPath lastPath;
};

class ZenoLinkFull final : public ZenoLink
{
public:
    ZenoLinkFull(ZenoSubGraphScene* pScene, const QString &fromId, const QString &fromPort, const QString &toId, const QString &toPort);

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

private:
    QString m_fromNodeid;
    QString m_fromPort;
    QString m_toNodeid;
    QString m_toPort;
    ZenoSubGraphScene* m_scene;
};

#endif