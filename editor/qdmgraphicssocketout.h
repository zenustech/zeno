#ifndef QDMGRAPHICSSOCKETOUT_H
#define QDMGRAPHICSSOCKETOUT_H

#include "qdmgraphicssocket.h"

ZENO_NAMESPACE_BEGIN

class QDMGraphicsSocketOut final : public QDMGraphicsSocket
{
public:
    QDMGraphicsSocketOut();

    virtual QPointF getLinkedPos() const override;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSOCKETOUT_H
