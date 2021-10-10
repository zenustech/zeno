#pragma once


#include <string>
#include <memory>
#include <zs/zeno/ztd/map.h>


namespace zs::zeno::dop {


struct Node;
struct Descriptor;


struct Graph {
    ztd::map<std::string, std::unique_ptr<Node>> nodes;

    Node *get_node(std::string const &name) const;
    Node *add_node(std::string const &name, Descriptor const &desc);
};


}
