#ifndef QDMGRAPHICSLINKFULL_H
#define QDMGRAPHICSLINKFULL_H

#include "qdmgraphicslink.h"
#include "qdmgraphicssocketin.h"
#include "qdmgraphicssocketout.h"

class QDMGraphicsLinkFull final : public QDMGraphicsLink
{
public:
    QDMGraphicsSocketOut *srcSocket;
    QDMGraphicsSocketIn *dstSocket;

    QDMGraphicsLinkFull(QDMGraphicsSocketOut *srcSocket, QDMGraphicsSocketIn *dstSocket);

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
};

#endif // QDMGRAPHICSLINKFULL_H
