#pragma once

#include <zeno/dop/Node.h>
#include <zeno/dop/SceneGraph.h>

ZENO_NAMESPACE_BEGIN
namespace dop {


struct SubnetIn;
struct SubnetOut;


struct SubnetNode : Node {
    std::unique_ptr<SceneGraph> subnet;
    SubnetIn *subnetIn;
    SubnetOut *subnetOut;

    virtual void apply() override;
};


struct SubnetIn : Node {
    SubnetNode *subnet{};

    virtual void apply() override;
};


struct SubnetOut : Node {
    SubnetNode *subnet{};

    virtual void apply() override;
};


}
ZENO_NAMESPACE_END
