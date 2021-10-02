#pragma once


#include "DopSocket.h"
#include "DopLazy.h"


struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;

    void apply_func();
    DopLazy get_output_by_name(std::string name);

    bool isvalid = false;

    void invalidate() {
        isvalid = false;
    }

    void serialize(std::ostream &ss) const;
};
