#pragma once


#include "DopSocket.h"


struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;
    bool applied = false;

    void apply_func();
    std::any get_output_by_name(std::string name);

    void invalidate() {
        applied = false;
    }

    void serialize(std::ostream &ss) const;
};
