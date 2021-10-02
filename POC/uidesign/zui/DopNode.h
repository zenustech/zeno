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

    bool node_changed = true; // deprecated

    void _apply_func(std::set<std::string> &visited);
    std::any get_output_by_name(std::string name, std::set<std::string> &visited);
    void serialize(std::ostream &ss) const;
    void invalidate();

    std::any get_input(int i, std::set<std::string> &visited);
    void set_output(int i, std::any val);
};
