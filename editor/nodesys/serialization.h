#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <rapidjson/document.h>
#include <vector>
#include "qdmgraphicsnode.h"
#include "qdmgraphicslinkfull.h"

ZENO_NAMESPACE_BEGIN

void serializeGraph
( rapidjson::Value &data
, rapidjson::Document::AllocatorType &alloc
, std::vector<QDMGraphicsNode *> const &nodes
, std::vector<QDMGraphicsLinkFull *> const &links
);

void deserializeGraph
( rapidjson::Value const &data
, std::vector<QDMGraphicsNode *> &nodes
, std::vector<QDMGraphicsLinkFull *> &links
);

ZENO_NAMESPACE_END

#endif // SERIALIZATION_H
