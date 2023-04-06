#pragma once

#include "HashMap.h"
#include "SimpleList.h"
#include "Pair.h"
#include "zeno/zeno.h"
#include <memory>
#include <string>

namespace zeno {
    ZENO_API std::string CopyToString(const char* data);
    ZENO_API char* CopyToChar(const std::string& str);

    ZENO_API SimpleList<std::pair<SimpleCharBuffer, zany>> CallTempNode(const std::shared_ptr<Graph>& graph, const char* id, size_t argc = 0, ...);
}

