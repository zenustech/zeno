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

    bool node_changed = true;

    void _apply_func();
    DopLazy get_output_by_name(std::string name, bool &changed);
    void serialize(std::ostream &ss) const;
    void invalidate();
};
