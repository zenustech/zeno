#pragma once


#include "stdafx.h"
#include "DopNode.h"


struct DopVisited {
    std::set<std::string> visited;

    bool is_visited(DopNode *node) {
        return node->isvalid && visited.contains(node->name);
    }

    void mark_visited(DopNode *node) {
        node->isvalid = true;
        visited.insert(node->name);
    }
};
