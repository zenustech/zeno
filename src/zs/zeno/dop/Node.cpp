#include <zs/zeno/dop/Node.h>
#include <zs/zeno/dop/execute.h>


namespace zs::zeno::dop {


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
