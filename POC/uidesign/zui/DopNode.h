#pragma once


#include "DopSocket.h"


struct DopVisited;
struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;

    void apply_func(DopVisited *visited);
    std::any get_output_by_name(DopVisited *visited, std::string name);

    bool isvalid = false;

    void invalidate() {
        isvalid = false;
    }

    void serialize(std::ostream &ss) const;
};
