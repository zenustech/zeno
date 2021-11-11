#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <zeno/rapidjson/rapidjson.h>
#include <vector>
#include "qdmgraphicsnode.h"
#include "qdmgraphicslinkfull.h"

ZENO_NAMESPACE_BEGIN

rapidjson::Value serializeGraph
( std::vector<QDMGraphicsNode *> const &nodes
, std::vector<QDMGraphicsLinkFull *> const &links
);

ZENO_NAMESPACE_END

#endif // SERIALIZATION_H
