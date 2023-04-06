#pragma once

#include "HashMap.h"
#include "SimpleList.h"
#include "Pair.h"
#include "zeno/zeno.h"
#include <memory>
#include <string>

namespace zeno {
    ZENO_API SimpleList<std::pair<SimpleCharBuffer, zany>> CallTempNode(Graph* graph, const char* id, size_t argc = 0, ...);
    ZENO_API Graph* AddSubnetNode(Graph* graph, const char* id);
    ZENO_API SimpleList<std::pair<SimpleCharBuffer, zany>> CallSubnetNode(Graph* graph, const char* id, size_t argc = 0, ...);
}

