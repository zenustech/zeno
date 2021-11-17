#ifndef QDMGRAPHICSLINKFULL_H
#define QDMGRAPHICSLINKFULL_H

#include "qdmgraphicslink.h"
#include "qdmgraphicssocketin.h"
#include "qdmgraphicssocketout.h"

ZENO_NAMESPACE_BEGIN

class QDMGraphicsLinkFull final : public QDMGraphicsLink
{
public:
    QDMGraphicsSocketOut *const srcSocket;
    QDMGraphicsSocketIn *const dstSocket;

    QDMGraphicsLinkFull(QDMGraphicsSocketOut *srcSocket, QDMGraphicsSocketIn *dstSocket);

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSLINKFULL_H
