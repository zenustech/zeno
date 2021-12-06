#include <zeno/dop/Node.h>
#include <zeno/dop/Executor.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


static std::string find_unique_name
    ( std::vector<std::string> const &names
    , std::string const &base
    )
{
    std::set<std::string> found;
    for (auto &&name: names) {
        if (name.starts_with(base)) {
            found.insert(name.substr(base.size()));
        }
    }

    if (found.empty())
        return base + '1';

    for (int i = 1; i <= found.size() + 1; i++) {
        std::string is = std::to_string(i);
        if (!found.contains(is)) {
            return base + is;
        }
    }

    return base + '0' + std::to_string(std::rand());
}


Node::Node() {
}


Node::~Node() = default;


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


Node *Node::linkInput(int idx, Node *node, int sockid) {
    inputs.touch(idx) = Input{.node = node, .sockid = sockid};
    return this;
}


ztd::any_ptr Node::getOutput(int idx) const {
    return outputs.touch(idx);
}



void Node::preapply(Executor *exec) {
    for (auto const &input: inputs) {
        exec->touch(input);
    }
}


}
ZENO_NAMESPACE_END
