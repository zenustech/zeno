#pragma once

#include <zeno/dop/Node.h>

ZENO_NAMESPACE_BEGIN
namespace dop {


struct SubnetIn;
struct SubnetOut;
struct Descriptor;


struct SubnetNode : Node {
    std::set<std::unique_ptr<Node>> subNodes;
    std::unique_ptr<SubnetIn> const subnetIn;
    std::unique_ptr<SubnetOut> const subnetOut;

    SubnetNode();
    ~SubnetNode() override;
    void apply() override;
    Node *addNode(Descriptor const &desc);
};


struct SubnetIn : Node {
    explicit SubnetIn(SubnetNode *subnet);
    ~SubnetIn() override;
    void apply() override;
};


struct SubnetOut : Node {
    explicit SubnetOut(SubnetNode *subnet);
    ~SubnetOut() override;
    void apply() override;
};


}
ZENO_NAMESPACE_END
