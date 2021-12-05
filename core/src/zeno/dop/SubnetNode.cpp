#include <zeno/dop/SubnetNode.h>
#include <zeno/dop/Descriptor.h>
#include <zeno/dop/Executor.h>
#include <zeno/dop/macros.h>


ZENO_NAMESPACE_BEGIN
namespace dop {

SubnetNode::SubnetNode()
    : subnetIn(std::make_unique<SubnetIn>(this))
    , subnetOut(std::make_unique<SubnetOut>(this))
{
}

SubnetNode::~SubnetNode() = default;

Node *SubnetNode::addNode(Descriptor const &desc) {
    auto node = desc.create();
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


SubnetIn::SubnetIn(SubnetNode *subnet)
    : subnet(subnet)
{}

SubnetIn::~SubnetIn() = default;

void SubnetIn::apply() {
}


SubnetOut::SubnetOut(SubnetNode *subnet)
    : subnet(subnet)
{}

SubnetOut::~SubnetOut() = default;

void SubnetOut::apply() {
}


}
ZENO_NAMESPACE_END
