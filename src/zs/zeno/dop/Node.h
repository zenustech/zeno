#pragma once


#include <zs/ztd/vector.h>
#include <zs/ztd/zany.h>
#include <variant>
#include <string>
#include <memory>
#include <vector>
#include <set>


namespace zs::zeno::dop {


struct Node;
struct Descriptor;


struct Input_Value {
    ztd::zany value;
};

struct Input_Link {
    Node *node = nullptr;
    int sockid = 0;
};

using Input = std::variant
< Input_Value
, Input_Link
>;


struct Node {
    ztd::vector<Input> inputs;
    ztd::vector<ztd::zany> outputs;

    float xpos = 0;
    std::string name;
    Descriptor *desc = nullptr;

    ztd::zany get_input(int idx) const;
    void set_output(int idx, ztd::zany val);
    
    template <class T>
    T get_input(int idx) const {
        return ztd::zany_cast<T>(get_input(idx));
    }

    virtual void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited);
    virtual void apply() = 0;
};


}
