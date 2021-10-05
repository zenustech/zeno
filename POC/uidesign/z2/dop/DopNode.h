#pragma once


#include <z2/dop/DopSocket.h>
#include <z2/dop/DopContext.h>


namespace z2::dop {


struct DopGraph;
struct DopNode;


struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;
    float xorder = 0;

    void _apply_func();
    std::any get_output_by_name(std::string sock_name);
    void serialize(std::ostream &ss) const;
    void invalidate();

    std::any get_input(int i) const;
    void set_output(int i, std::any val);
};


}  // namespace z2::dop
