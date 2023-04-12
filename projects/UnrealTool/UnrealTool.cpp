#include "zeno/unreal/UnrealTool.h"
#include "msgpack/msgpack.h"
#include <cstdarg>
#include <map>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/PrimitiveObject.h>

#define PROC_INPUT_ARGS                                                      \
    typedef std::pair<const char *, zeno::zany> ZPair;                       \
                                                                             \
    if (nullptr == graph) {                                                  \
        return {0};                                                          \
    }                                                                        \
                                                                             \
    std::map<std::string, zeno::zany> inputs;                                \
                                                                             \
    va_list vl;                                                              \
    va_start(vl, argc);                                                      \
    for (size_t i = 0; i < argc; ++i) {                                      \
        const ZPair &pair = va_arg(vl, ZPair);                               \
        inputs.insert(std::make_pair(std::string(pair.first), pair.second)); \
    }                                                                        \
    va_end(vl);

#define CONVERT_SIMPLE_LIST_OUTPUTS                                                                             \
    zeno::SimpleList<std::pair<zeno::SimpleCharBuffer, zeno::zany>> result(output.size());                      \
    for (const auto &pair : output) {                                                                           \
        result.add(std::make_pair(zeno::SimpleCharBuffer{pair.first.c_str(), pair.first.size()}, pair.second)); \
    }                                                                                                           \
                                                                                                                \
    return result;

zeno::SubnetNode* GetSubnetNodeToCall(zeno::Graph* graph) {

    if (nullptr != graph && !(graph->nodesToExec.empty())) {
        const std::string& execNodeNameExt = *graph->nodesToExec.begin();
        const auto splitPos = execNodeNameExt.find(':');
        const std::string execNodeName = execNodeNameExt.substr(0, splitPos);
        // Check node is a subnet node
        if (graph->nodes.find(execNodeName) != graph->nodes.end()) {
            auto* execNode = dynamic_cast<zeno::SubnetNode*>(graph->nodes[execNodeName].get());
            return execNode;
        }
    }

    return nullptr;
}

zeno::SimpleList<std::pair<zeno::SimpleCharBuffer, zeno::zany>> zeno::CallTempNode(Graph *graph, const char *id,
                                                                                   size_t argc, ...) {
    PROC_INPUT_ARGS;

    std::map<std::string, std::shared_ptr<zeno::IObject>> output = graph->callTempNode(id, inputs);

    CONVERT_SIMPLE_LIST_OUTPUTS;
}

zeno::Graph *zeno::AddSubnetNode(Graph *graph, const char *id) {
    if (nullptr != graph) {
        return graph->addSubnetNode(id);
    }
    return nullptr;
}

zeno::SimpleList<std::pair<zeno::SimpleCharBuffer, zeno::zany>> zeno::CallSubnetNode(zeno::Graph *graph, const char *id,
                                                                                     size_t argc, ...) {
    PROC_INPUT_ARGS;

    std::map<std::string, zany> output = graph->callSubnetNode(id, std::move(inputs));

    CONVERT_SIMPLE_LIST_OUTPUTS;
}

bool zeno::LoadGraphChecked(zeno::Graph *graph, const char *json) {
    if (nullptr == graph || nullptr == json) return false;
    try {
        graph->loadGraph(json);
    } catch (...) {
        return false;
    }
    return true;
}

bool zeno::IsValidZSL(const char *json) {
    auto graph = zeno::getSession().createGraph();
    if (!graph) return false;
    if (!LoadGraphChecked(graph.get(), json)) return false;
    // Check have nodes to exec
    if (graph->nodesToExec.empty()) return false;

    const zeno::SubnetNode* execNode = GetSubnetNodeToCall(graph.get());
    if (nullptr == execNode || !execNode->subgraph) return false;

    return true;
}

zeno::SimpleCharBuffer zeno::CallSubnetNode_Mesh(zeno::Graph *graph, const char *id, size_t argc, ...) {
    PROC_INPUT_ARGS;

    std::map<std::string, zany> output = graph->callSubnetNode(id, std::move(inputs));

    for (const auto& pair : output) {
        if (!pair.second) {
            continue;
        }
        zeno::IObject* obj = pair.second.get();
        auto* primitiveObject = dynamic_cast<zeno::PrimitiveObject*>(obj);
        if (nullptr == primitiveObject) {
            continue;
        }

        zeno::unreal::Mesh mesh { primitiveObject->verts, primitiveObject->tris };
        auto res = msgpack::pack(mesh);
        return SimpleCharBuffer { reinterpret_cast<char*>(res.data()), res.size() };
    }

    return nullptr;
}

zeno::SimpleCharBuffer zeno::GetGraphInputParams(zeno::Graph *graph) {
    zeno::unreal::SubnetNodeParamList list;

    if (nullptr != graph) {
        const zeno::SubnetNode* execNode = GetSubnetNodeToCall(graph);
        if (execNode != nullptr && execNode->subgraph) {
            for (const auto& inputNode : execNode->subgraph->subInputNodes) {
                auto& info = execNode->subgraph->nodes[inputNode.second];
                zeno::StringObject* typeObj = dynamic_cast<zeno::StringObject*>(info->inputs["type:"].get());
                if (typeObj) {
                    list.params.insert(std::make_pair(inputNode.first, (int8_t)zeno::unreal::ConvertStringToEParamType(typeObj->value)));
                }
            }
        }
    }

    auto data = msgpack::pack(list);
    return { reinterpret_cast<char*>(data.data()), data.size() };
}
