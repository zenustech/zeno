#pragma once


#include <zeno/ztd/vector.h>
#include <zeno/ztd/zany.h>
#include <variant>
#include <string>
#include <memory>
#include <vector>
#include <set>


namespace dop {


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
    float ypos = 0;
    Descriptor const *desc = nullptr;

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
ZENO_NAMESPACE_END
