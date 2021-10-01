#pragma once


#include "DopSocket.h"
#include "DopVisited.h"


struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;
    bool applied = false;

    void apply_func(DopVisited *visited);
    std::any get_output_by_name(DopVisited *visited, std::string name);

    void invalidate() {
        applied = false;
    }

    void serialize(std::ostream &ss) const;
};
