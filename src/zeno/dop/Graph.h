#pragma once


#include <string>
#include <memory>
#include <vector>
#include <zeno/common.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct Node;
struct Descriptor;


struct Graph {
    std::vector<std::unique_ptr<Node>> nodes;

    Node *add_node(Descriptor const &desc);
};


}
ZENO_NAMESPACE_END
