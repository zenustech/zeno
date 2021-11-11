#include "serialization.h"

ZENO_NAMESPACE_BEGIN

void serializeGraph
( rapidjson::Value &data
, rapidjson::Document::AllocatorType &alloc
, std::vector<QDMGraphicsNode *> const &nodes
, std::vector<QDMGraphicsLinkFull *> const &links
) {
    data.SetInt(42);
}

void deserializeGraph
( rapidjson::Value const &data
, std::vector<QDMGraphicsNode *> &nodes
, std::vector<QDMGraphicsLinkFull *> &links
) {
}

ZENO_NAMESPACE_END
