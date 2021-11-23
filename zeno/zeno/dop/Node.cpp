#include <zeno/dop/Node.h>
#include <zeno/dop/Executor.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::any_ptr Node::get_input(int idx) const {
    return Executor::getval(inputs.at(idx));
}


void Node::set_output(int idx, ztd::any_ptr val) {
    outputs.at(idx) = std::move(val);
}


void Node::preapply(Executor *exec, std::vector<Node *> &tolink, std::set<Node *> &visited) {
    for (auto node: inputs) {
        exec->touch(node, tolink, visited);
    }
    tolink.push_back(this);
}


}
ZENO_NAMESPACE_END
