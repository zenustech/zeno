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

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    ZENO_API SubnetNode();
    ZENO_API ~SubnetNode();

    void init(const NodeData& dat);

    ZENO_API std::vector<std::shared_ptr<IParam>> get_input_params() const override;
    ZENO_API std::vector<std::shared_ptr<IParam>> get_output_params() const override;
    ZENO_API params_change_info update_editparams(const std::vector<std::pair<zeno::ParamInfo, std::string>>& params) override;
    ZENO_API void add_param(bool bInput, const ParamInfo& param);
    ZENO_API void remove_param(bool bInput, const std::string& name);

    ZENO_API virtual void apply() override;
};

struct ImplSubnetNodeClass : INodeClass {
    ImplSubnetNodeClass() : INodeClass({}, "") {
    }

    virtual std::shared_ptr<INode> new_instance(std::string const& name) const override {
        std::shared_ptr<SubnetNode> spNode = std::make_shared<SubnetNode>();

        //TODO: need to find descriptors name, to create Subnet.

        spNode->set_name(name);
        spNode->m_nodecls = classname;
        return spNode;
    }
};

}
