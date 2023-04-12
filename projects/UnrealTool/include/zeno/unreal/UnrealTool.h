#pragma once

#include "HashMap.h"
#include "SimpleList.h"
#include "Pair.h"
#include "ZenoUnrealTypes.h"
#include "zeno/zeno.h"
#include <memory>
#include <string>

namespace zeno {
    /**
     * Wrapper of Graph::callTempNode()
     * @param graph graph to call with
     * @param id node class name
     * @param argc number of your inputs
     * @param ... vargs with type std::pair<const char*, zany>
     * @return list of result
     */
    ZENO_API SimpleList<std::pair<SimpleCharBuffer, zany>> CallTempNode(Graph* graph, const char* id, size_t argc = 0, ...);
    /**
     * Wrapper of Graph::addSubnetNode()
     * @param graph graph to call with
     * @param id node name
     * @return graph created
     */
    ZENO_API Graph* AddSubnetNode(Graph* graph, const char* id);
    /**
     * @deprecated Use specialized function such as ['GetGraphInputParams']
     * @param graph graph to call with
     * @param id node class name
     * @param argc number of your inputs
     * @param ... vargs with type std::pair<const char*, zany>
     * @return list of result
     */
    ZENO_API SimpleList<std::pair<SimpleCharBuffer, zany>> CallSubnetNode(Graph* graph, const char* id, size_t argc = 0, ...);
    /**
     * Wrapper of Graph::loadGraph()
     * @param graph graph to call with
     * @param json zsl content
     * @return false if any exception
     */
    ZENO_API bool LoadGraphChecked(Graph* graph, const char* json);
    /**
     * Check string is a zsl
     * @param json zsl file to be valid
     * @return valid?
     */
    ZENO_API bool IsValidZSL(const char* json);
    /**
     * Apply subnet node and return json mesh data
     * @param graph graph to call with
     * @param id subnet node name
     * @param argc number of your inputs
     * @param ... vargs with type std::pair<const char*, zany>
     * @return ['Mesh']
     */
    ZENO_API SimpleCharBuffer CallSubnetNode_Mesh(Graph* graph, const char* id, size_t argc = 0, ...);
    /**
     * Get the input params required by node to exec which marked in zeno editor.
     * @param graph graph to call with
     * @return ['SubnetNodeParamList']
     */
    ZENO_API SimpleCharBuffer GetGraphInputParams(Graph* graph);
}
