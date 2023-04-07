#include "zeno/unreal/UnrealTool.h"
#include "zeno/unreal/HashMap.h"
#include "zeno/unreal/Pair.h"
#include "zeno/unreal/SimpleList.h"
#include <cstdarg>
#include <map>
#include <memory.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>

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

    const std::string& execNodeNameExt = *graph->nodesToExec.begin();
    const auto splitPos = execNodeNameExt.find(':');
    const std::string execNodeName = execNodeNameExt.substr(0, splitPos);
    // Check node is a subnet node
    if (graph->nodes.find(execNodeName) == graph->nodes.end()) return false;
    const zeno::SubnetNode* execNode = dynamic_cast<zeno::SubnetNode*>(graph->nodes[execNodeName].get());
    if (nullptr == execNode || !execNode->subgraph) return false;

    return true;
}
