#pragma once


#include <z2/ztd/vector.h>
#include <variant>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <any>


namespace z2::dop {


struct Node;
struct Descriptor;


struct Input_Value {
    std::any value;
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
    ztd::vector<std::any> outputs;

    float xpos = 0;
    std::string name;
    Descriptor *desc = nullptr;

    std::any get_input(int idx) const;
    std::any get_input(std::string const &name) const;
    void set_output(int idx, std::any val);
    
    template <class T>
    T get_input(int idx) const {
        return std::any_cast<T>(get_input(idx));
    }

    template <class T>
    T get_input(std::string const &name) const {
        return std::any_cast<T>(get_input(name));
    }

    virtual void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited);
    virtual void apply() = 0;
};


}
