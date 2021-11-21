#pragma once

#include <zeno/dop/Node.h>

ZENO_NAMESPACE_BEGIN
namespace dop {


struct SubnetIn;
struct SubnetOut;


struct SubnetNode : Node {
    std::vector<SubnetIn *> subins;
    std::vector<SubnetOut *> subouts;

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
