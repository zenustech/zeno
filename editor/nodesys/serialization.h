#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <zeno/common.h>
#include <vector>
#include "qdmgraphicsnode.h"
#include "qdmgraphicslinkfull.h"

ZENO_NAMESPACE_BEGIN

void serializeGraph
( std::vector<QDMGraphicsNode *> const &nodes
, std::vector<QDMGraphicsLinkFull *> const &links
);

ZENO_NAMESPACE_END

#endif // SERIALIZATION_H
