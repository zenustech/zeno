#pragma once


#include <string>
#include <memory>
#include <vector>


namespace zeno::dop {


struct Node;
struct Descriptor;


struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;

    Node *add_node(Descriptor const &desc);
};


}
