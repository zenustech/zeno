#ifndef QDMGRAPHICSSOCKETIN_H
#define QDMGRAPHICSSOCKETIN_H

#include "qdmgraphicssocket.h"

ZENO_NAMESPACE_BEGIN

class QDMGraphicsSocketIn final : public QDMGraphicsSocket
{
public:
    QDMGraphicsSocketIn();

    virtual void unlinkAll() override;
    virtual void linkRemoved(QDMGraphicsLinkFull *link) override;
    virtual void linkAttached(QDMGraphicsLinkFull *link) override;
    virtual QPointF getLinkedPos() const override;
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSOCKETIN_H
