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


std::any getval(Input const &input);
std::any resolve(Input const &input, std::set<Node *> &visited);
void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
void sortexec(Node *root, std::vector<Node *> &tolink, std::set<Node *> &visited);


inline std::any resolve(Input const &input) {
    std::set<Node *> visited;
    return resolve(input, visited);
}


struct Node {
    ztd::vector<Input> inputs;
    ztd::vector<std::any> outputs;

    float xpos = 0;
    std::string name;

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
