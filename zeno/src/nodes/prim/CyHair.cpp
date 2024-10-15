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

        auto path = get_input2<std::string>("path");
        bool exist = std::filesystem::exists(path);
        bool yup = get_input2<bool>("yup");
        
        if (!exist) {
            throw std::string("CyHair file doesn't exist");
        }

        auto out = std::make_shared<zeno::PrimitiveObject>();
        out->userData().set2("yup", yup);
        out->userData().set2("path", path);
        out->userData().set2("cyhair", true);
    
        set_output("out", std::move(out));
    }
};

ZENDEFNODE(CyHair, 
{   {   
        {gParamType_String, "path", "", Socket_Primitve, ReadPathEdit},
        {gParamType_Bool, "yup", "1"},
    },
    {{gParamType_Primitive, "out"}}, //output
    {},
    {"read"}
});

} // namespace