#include <z2/dop/Node.h>
#include <z2/dop/Descriptor.h>
#include <z2/dop/execute.h>


namespace z2::dop {


std::any Node::get_input(int idx) const {
    return getval(inputs.at(idx));
}


std::any Node::get_input(std::string const &name) const {
    for (int i = 0; i < desc->inputs.size(); i++) {
        if (desc->inputs[i].name == name) {
            return get_input(i);
        }
    }
    return {};
}


void Node::set_output(int idx, std::any val) {
    outputs.at(idx) = std::move(val);
}


void Node::preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) {
    for (auto node: inputs) {
        touch(node, tolink, visited);
    }
    tolink.push_back(this);
}


}
