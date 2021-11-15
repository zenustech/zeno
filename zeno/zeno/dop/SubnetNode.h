#pragma once

#include <zeno/dop/Node.h>

ZENO_NAMESPACE_BEGIN
namespace dop {


struct SubnetNode : Node {
    std::vector<Node *> outputs;

    virtual void apply() override;
};


}
ZENO_NAMESPACE_END
