#pragma once

#include <unordered_set>
#include <zeno/core/common.h>
#include <mutex>

namespace zeno {

struct CalcContext {
    //代表一条计算链路下的上下文缓存记录，起点通常是view的节点

    std::unordered_set<std::string> uuid_node_params;
    std::unordered_set<std::string> visited_nodes;
    int curr_iter = -1;     //用于调试

    std::mutex mtx;
};



}