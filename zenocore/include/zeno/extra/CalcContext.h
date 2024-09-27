#pragma once

#include <unordered_set>

namespace zeno {

struct CalcContext {
    std::unordered_set<std::string> uuid_node_params;
    std::unordered_set<std::string> visited_nodes;
};

}