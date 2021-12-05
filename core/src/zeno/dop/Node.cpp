#include <zeno/dop/Node.h>
#include <zeno/dop/Executor.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::any_ptr Node::get_input(int idx) const {
    return Executor::getval(inputs.touch(idx));
}


void Node::set_output(int idx, ztd::any_ptr val) {
    outputs.touch(idx) = std::move(val);
}


Node *Node::setInput(int idx, ztd::any_ptr val) {
    inputs.touch(idx) = Input{.value = val};
    return this;
}


Node *Node::setInput(int idx, Node *node, int sockid) {
    inputs.touch(idx) = Input{.node = node, .sockid = sockid};
    return this;
}


ztd::any_ptr Node::getOutput(int idx) const {
    return outputs.touch(idx);
}



void Node::preapply(Executor *exec) {
    for (auto node: inputs) {
        exec->touch(node);
    }
}


}
ZENO_NAMESPACE_END
