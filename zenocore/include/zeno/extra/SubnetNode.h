#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>

namespace zeno {

struct SubnetNode : INode {
    std::unique_ptr<INodeClass> subnetClass;
    //std::vector<std::string> inputKeys;
    //std::vector<std::string> outputKeys;
    std::shared_ptr<Graph> const subgraph;

    ZENO_API SubnetNode();
    ZENO_API ~SubnetNode();

    void init(const NodeData& dat);

    //void addSubnetInput(std::string const &key) {
        //subnetClass->desc->inputs.push_back({{}, key, {}});
        //inputKeys.push_back(key);
    //}

    //void addSubnetOutput(std::string const &key) {
        //subnetClass->desc->outputs.push_back({{}, key, {}});
        //outputKeys.push_back(key);
    //}

    ZENO_API virtual void apply() override;
};

struct ImplSubnetNodeClass : INodeClass {
    ImplSubnetNodeClass() : INodeClass({}, "") {
    }

    virtual std::shared_ptr<INode> new_instance(std::string const& name) const override {
        std::shared_ptr<SubnetNode> spNode = std::make_shared<SubnetNode>();

        //TODO: need to find descriptors name, to create Subnet.

        spNode->name = name;
        spNode->nodecls = classname;
        return spNode;
    }
};

}
