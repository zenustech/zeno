#pragma once


#include <cstdio>
#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <set>
#include <any>


namespace z2::dop {


struct Node;


using Input = std::variant<std::any, Node *>;


struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;
};


std::any getval(Input const &input);
std::any resolve(Input const &input, std::set<Node *> &visited);
void touch(Input const &input, std::vector<Node *> &tolink, std::set<Node *> &visited);
void sortexec(std::vector<Node *> &tolink, std::set<Node *> &visited);


struct Node {
    float xorder = 0;
    std::vector<Input> inputs;
    std::any result;

    std::any get_input(int idx) const;

    template <class T>
    T get_input(int idx) const {
        return std::any_cast<T>(get_input(idx));
    }

    void set_output(int idx, std::any val);

    virtual void preapply(std::vector<Node *> &tolink, std::set<Node *> &visited);
    virtual void apply() = 0;
};


}
