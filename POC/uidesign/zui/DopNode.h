#pragma once


#include "DopLazy.h"
#include "DopSocket.h"


struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;

    void apply_func();
    DopLazy get_output_by_name(std::string name);
    void serialize(std::ostream &ss) const;
    void invalidate();
};
