#pragma once
#include<zeno/zeno.h>
struct ReadUSD : zeno::INode {
    virtual void apply() override;
};
struct ImportUSDMesh : zeno::INode {
    virtual void apply() override;
};
struct ImportUSDPrimMatrix : zeno::INode {
    virtual void apply() override;
};
struct ViewUSDTree : zeno::INode {
    int _getDepth(const std::string& primPath) const;
    virtual void apply() override;
};
struct USDShowAllPrims : zeno::INode {
    virtual void apply() override;
};
struct ShowPrimUserData : zeno::INode {
    virtual void apply() override;
};
struct ShowUSDPrimAttribute : zeno::INode {
    void _showAttribute(std::any, bool) const;
    virtual void apply() override;
};
struct ShowUSDPrimRelationShip : zeno::INode {
    virtual void apply() override;
};
struct EvalUSDPrim : zeno::INode {
    virtual void apply() override;
};