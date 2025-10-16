#pragma once

#include <zeno/core/NodeImpl.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Descriptor.h>
#include <zeno/utils/uuid.h>

namespace zeno {

struct ZENO_API SubnetNode : NodeImpl {

    SubnetNode(INode* pNode);
    ~SubnetNode();

    void apply() override;
    float time() const;

    void initParams(const NodeData& dat) override;
    params_change_info update_editparams(const ParamsUpdateInfo& params, bool bSubnetInit = false) override;
    Graph* get_subgraph() const;
    void init_graph(std::shared_ptr<Graph> subg);
    bool isAssetsNode() const;
    bool is_loaded() const override;
    NodeType nodeType() const override;
    void mark_clean() override;

    void cleanInternalCaches();

    NodeData exportInfo() const override;
    CustomUI get_customui() const override;
    CustomUI export_customui() const override;

    void setCustomUi(const CustomUI& ui);
    void mark_subnetdirty(bool bOn);

    bool is_locked() const override;
    void set_locked(bool bLocked) override;
    CALLBACK_REGIST(lockChanged, void, void)

    bool is_clearsubnet() const;
    void set_clearsubnet(bool bOn);
    CALLBACK_REGIST(clearSubnetChanged, void, bool)

protected:
    CustomUI m_customUi;
    std::shared_ptr<Graph> m_subgraph;    //assets subnetnode

private:
    bool m_bLocked; //只给资产用
    bool m_bClearSubnet;    //计算后清除子图节点缓存
};

}
