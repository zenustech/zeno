#ifndef QDMGRAPHICSLINKHALF_H
#define QDMGRAPHICSLINKHALF_H

#include "qdmgraphicslink.h"
#include "qdmgraphicssocket.h"

class QDMGraphicsLinkHalf final : public QDMGraphicsLink
{
    QPointF getMousePos() const;

public:
    QDMGraphicsSocket *socket;

    QDMGraphicsLinkHalf(QDMGraphicsSocket *socket);

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
};

#endif // QDMGRAPHICSLINKHALF_H
