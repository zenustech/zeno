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

private:
    std::string _allocateNodeName(std::string const &base);
};


struct SubnetIn : Node {
    SubnetIn();
    ~SubnetIn() override;
    void apply() override;
};


struct SubnetOut : Node {
    SubnetOut();
    ~SubnetOut() override;
    void apply() override;
};


}
ZENO_NAMESPACE_END
