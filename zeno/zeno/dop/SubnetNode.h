#pragma once

#include <zeno/dop/Node.h>

ZENO_NAMESPACE_BEGIN
namespace dop {


struct SubnetNode : Node {
    std::vector<Node *> subins;
    std::vector<Node *> subouts;

    virtual void apply() override;
};


}
ZENO_NAMESPACE_END
