#include "qdmgraphicssocketin.h"
#include "qdmgraphicsnode.h"
#include "qdmgraphicslinkfull.h"

ZENO_NAMESPACE_BEGIN

QDMGraphicsSocketIn::QDMGraphicsSocketIn()
{
    label->setPos(SIZE / 2, -SIZE * 0.7f);
}

QPointF QDMGraphicsSocketIn::getLinkedPos() const
{
    return scenePos() - QPointF(SIZE / 2, 0);
}

ZENO_NAMESPACE_END
