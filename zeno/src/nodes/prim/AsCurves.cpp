#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/CurveType.h>
#include <zeno/types/UserData.h>
#include <zeno/zeno.h>

#include "magic_enum.hpp"

namespace zeno {

struct AsCurves : zeno::INode {
    virtual void apply() override {
        
        auto prim = get_input2<zeno::PrimitiveObject>("prim");

        auto typeString = get_input2<std::string>("type:");
        auto typeEnum = magic_enum::enum_cast<CurveType>(typeString).value_or(CurveType::LINEAR);
        auto typeIndex = (int)magic_enum::enum_index<CurveType>(typeEnum).value_or(0);

        prim->userData().set2("curve", typeIndex);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(AsCurves, 
{   {   
        {gParamType_Primitive, "prim"},
    },
    {{gParamType_Primitive, "prim"}}, //output
    {
        {"enum " + zeno::CurveTypeListString(), "type", zeno::CurveTypeDefaultString() }
    },            //prim
    {"prim"}
});

} // namespace