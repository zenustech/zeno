#include <zeno2/dop/Node.h>
#include <zeno2/dop/execute.h>


namespace zeno2::dop {


ztd::zany Node::get_input(int idx) const {
    return getval(inputs.at(idx));
}


void Node::set_output(int idx, ztd::zany val) {
    outputs.at(idx) = std::move(val);
}


void Node::preapply(std::vector<Node *> &tolink, std::set<Node *> &visited) {
    for (auto node: inputs) {
        touch(node, tolink, visited);
    }
    tolink.push_back(this);
}


}
