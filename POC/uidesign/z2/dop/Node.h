#pragma once


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


struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;

    Node *add_node(Descriptor const &desc);
};


std::any getval(Input const &input);
std::any resolve(Input const &input, std::set<Node *> &visited);
void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
void sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited);


struct Node {
    float xorder = 0;
    std::vector<Input> inputs;
    std::vector<std::any> outputs;

    std::any get_input(int idx) const;
    void set_output(int idx, std::any val);

    template <class T>
    T get_input(int idx) const {
        return std::any_cast<T>(get_input(idx));
    }

    virtual void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited);
    virtual void apply() = 0;
};


}
