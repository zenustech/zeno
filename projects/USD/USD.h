#pragma once
#include<zeno/zeno.h>

#ifdef _MSC_VER
    #if defined(DLLEXPORT)
        #define USD_API __declspec(dllexport)
    #else
        #define USD_API __declspec(dllimport)
    #endif
#else
    #define USD_API
#endif

struct ReadUSD : zeno::INode {
    USD_API virtual void apply() override;
};
struct ImportUSDMesh : zeno::INode {
    USD_API virtual void apply() override;
};
struct ImportUSDPrimMatrix : zeno::INode {
    USD_API virtual void apply() override;
};
struct ViewUSDTree : zeno::INode {
    USD_API int _getDepth(const std::string& primPath) const;
    USD_API virtual void apply() override;
};
struct USDShowAllPrims : zeno::INode {
    USD_API virtual void apply() override;
};
struct ShowPrimUserData : zeno::INode {
    USD_API virtual void apply() override;
};
struct ShowUSDPrimAttribute : zeno::INode {
    USD_API void _showAttribute(std::any, bool) const;
    USD_API virtual void apply() override;
};
struct ShowUSDPrimRelationShip : zeno::INode {
    USD_API virtual void apply() override;
};
struct EvalUSDPrim : zeno::INode {
    USD_API virtual void apply() override;
};