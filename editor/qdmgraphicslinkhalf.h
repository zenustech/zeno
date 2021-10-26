#ifndef QDMGRAPHICSLINKHALF_H
#define QDMGRAPHICSLINKHALF_H

#include "qdmgraphicslink.h"
#include "qdmgraphicssocket.h"

ZENO_NAMESPACE_BEGIN

class QDMGraphicsLinkHalf final : public QDMGraphicsLink
{
public:
    QDMGraphicsSocket *socket;

    QDMGraphicsLinkHalf(QDMGraphicsSocket *socket);

    virtual QPointF getSrcPos() const override;
    virtual QPointF getDstPos() const override;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSLINKHALF_H
