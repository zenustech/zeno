#ifndef QDMGRAPHICSSOCKETOUT_H
#define QDMGRAPHICSSOCKETOUT_H

#include "qdmgraphicssocket.h"

class QDMGraphicsSocketOut final : public QDMGraphicsSocket
{
public:
    QDMGraphicsSocketOut();

    virtual QPointF getLinkedPos() const override;
};

#endif // QDMGRAPHICSSOCKETOUT_H
