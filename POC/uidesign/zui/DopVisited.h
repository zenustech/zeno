#pragma once


#include "stdafx.h"


struct DopVisited {
    std::set<std::string> visited;

    inline bool is_visited(std::string const &name) {
        return visited.contains(name);
    }
};
