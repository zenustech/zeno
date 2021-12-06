#include <zeno/dop/SubnetNode.h>
#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Executor.h>
#include <zeno/dop/macros.h>


ZENO_NAMESPACE_BEGIN
namespace dop {

std::string SubnetNode::_allocateNodeName(std::string const &base)
{
    std::set<std::string> found;
    for (auto const &node: subNodes) {
        auto const &name = node->name;
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

SubnetNode::SubnetNode()
    : subnetIn(std::make_unique<SubnetIn>())
    , subnetOut(std::make_unique<SubnetOut>())
{
    subnetIn->subnet = this;
    subnetIn->name = "SubnetIn";
    subnetOut->subnet = this;
    subnetOut->name = "SubnetOut";
}

SubnetNode::~SubnetNode() = default;

Node *SubnetNode::addNode(Descriptor const &desc) {
    auto node = desc.create();
    node->subnet = this;
    node->name = _allocateNodeName(desc.name);

    auto node_ptr = node.get();
    subNodes.insert(std::move(node));
    return node_ptr;
}

void SubnetNode::apply() {
    for (int i = 0; i < inputs.size(); i++) {
        subnetIn->set_output(i, get_input(i));
    }

    Executor exec;
    for (int i = 0; i < subnetOut->inputs.size(); i++) {
        set_output(i, exec.resolve(subnetOut->inputs[i]));
    }
}

ZENO_DOP_DEFCLASS(SubnetNode, {{
    "misc", "a custom subnet to combine many nodes into one",
}, {
}, {
}});


SubnetIn::SubnetIn() = default;
SubnetIn::~SubnetIn() = default;

void SubnetIn::apply() {
}


SubnetOut::SubnetOut() = default;
SubnetOut::~SubnetOut() = default;

void SubnetOut::apply() {
}


}
ZENO_NAMESPACE_END
