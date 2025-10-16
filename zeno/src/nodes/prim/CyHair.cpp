#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/CurveType.h>
#include <zeno/types/UserData.h>
#include <zeno/zeno.h>

#include "magic_enum.hpp"

#include <filesystem>
#include <vector>

namespace zeno {

struct CyHair : zeno::INode {
    virtual void apply() override {

        auto path = ZImpl(get_input2<std::string>("path"));
        bool exist = std::filesystem::exists(path);
        bool yup = ZImpl(get_input2<bool>("yup"));
        
        if (!exist) {
            throw std::string("CyHair file doesn't exist");
        }

        auto out = std::make_unique<zeno::PrimitiveObject>();
        out->userData()->set_bool("yup", yup);
        out->userData()->set_string("path", stdString2zs(path));
        out->userData()->set_bool("cyhair", true);
    
        ZImpl(set_output("out", std::move(out)));
    }
};

ZENDEFNODE(CyHair, 
{   {   
        {gParamType_String, "path", "", Socket_Primitve, ReadPathEdit},
        {gParamType_Bool, "yup", "1"},
    },
    {
        {gParamType_Primitive, "out"}
    }, //output
    {},
    {"read"}
});

} // namespace