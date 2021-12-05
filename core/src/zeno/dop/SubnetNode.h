#pragma once

#include <zeno/dop/Node.h>
#include <zeno/dop/SceneGraph.h>

ZENO_NAMESPACE_BEGIN
namespace dop {


struct SubnetIn;
struct SubnetOut;


struct SubnetNode : Node {
    std::set<std::unique_ptr<Node>> subNodes;
    std::unique_ptr<SubnetIn> subnetIn;
    std::unique_ptr<SubnetOut> subnetOut;

    SubnetNode();
    ~SubnetNode() override;
    void apply() override;
};


struct SubnetIn : Node {
    SubnetNode *subnet{};

    explicit SubnetIn(SubnetNode *subnet);
    ~SubnetIn() override;
    void apply() override;
};


struct SubnetOut : Node {
    SubnetNode *subnet{};

    explicit SubnetOut(SubnetNode *subnet);
    ~SubnetOut() override;
    void apply() override;
};


}
ZENO_NAMESPACE_END
