#include "serialization.h"
#include <zeno/zmt/log.h>

ZENO_NAMESPACE_BEGIN


void serializeGraph
( rapidjson::Value &data
, rapidjson::Document::AllocatorType &alloc
, std::vector<QDMGraphicsNode *> const &nodes
, std::vector<QDMGraphicsLinkFull *> const &links
) {

    data.SetObject();

    rapidjson::Value d_nodes(rapidjson::kArrayType);
    for (auto node: nodes) {
        rapidjson::Value d_node(rapidjson::kObjectType);

        auto name = node->getName().toStdString();
        rapidjson::Value d_name;
        d_name.SetString(name.data(), name.size());
        ZENO_LOG_INFO("! {}", name.data());
        d_node.AddMember("name", d_name, alloc);

        d_nodes.PushBack(d_node, alloc);
    }

    data.AddMember("nodes", d_nodes, alloc);
}


void deserializeGraph
( rapidjson::Value const &data
, std::vector<QDMGraphicsNode *> &nodes
, std::vector<QDMGraphicsLinkFull *> &links
) {
}


ZENO_NAMESPACE_END
