#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/Session.h>

namespace zeno {

struct SubnetNode : INode {
    std::unique_ptr<INodeClass> subnetClass;

    ZENO_API virtual void apply() override;
};

struct ImplSubnetNodeClass : INodeClass {
    ImplSubnetNodeClass(Descriptor const &desc) : INodeClass(desc) {
    }

    virtual std::unique_ptr<INode> new_instance() const override {
        return std::make_unique<SubnetNode>();
    }
};

}
