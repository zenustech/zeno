#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>

namespace zeno {

struct SubnetNode : INode {
    std::shared_ptr<Graph> subgraph;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;

    CustomUI m_customUi;

    ZENO_API SubnetNode();
    ZENO_API ~SubnetNode();

    ZENO_API void initParams(const NodeData& dat) override;
    ZENO_API params_change_info update_editparams(const ParamsUpdateInfo& params) override;
    ZENO_API std::shared_ptr<Graph> get_graph() const;
    ZENO_API bool isAssetsNode() const;
    ZENO_API void apply() override;
    ZENO_API NodeData exportInfo() const override;

    ZENO_API CustomUI get_customui() const override;
    ZENO_API void setCustomUi(const CustomUI& ui);
    void mark_subnetdirty(bool bOn);
};

}
