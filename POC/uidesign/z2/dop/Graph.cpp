#include <z2/dop/Graph.h>
#include <z2/dop/Node.h>
#include <z2/dop/Descriptor.h>


namespace z2::dop {


Node *Graph::get_node(std::string const &name) const {
    return nodes.at(name).get();
}


Node *Graph::add_node(std::string const &name, Descriptor const &desc) {
    auto node = desc.factory();
    node->inputs.resize(desc.inputs.size());
    node->outputs.resize(desc.outputs.size());
    auto p = node.get();
    nodes.emplace(name, std::move(node));
    return p;
}


}
