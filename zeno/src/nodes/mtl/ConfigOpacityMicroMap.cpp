#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/MaterialObject.h>

#include <stdexcept>
#include <filesystem>
#include <tinygltf/json.hpp>

namespace zeno {

    struct ConfigOpacityMicroMap : INode {

    virtual void apply() override {
        auto path = zsString2Std(get_input2_string("path"));
        bool exist = std::filesystem::exists(path);
        if (!exist) {
            throw std::runtime_error("File " + path + " doesn't exist");
        }
        nlohmann::json ommj;
        ommj["path"] = path;
        ommj["alphaMode"] = zsString2Std(get_input2_string("alphaMode"));
        ommj["opacityCutoff"] = get_input2_float("opacityCutoff");
        ommj["transparencyCutoff"] = get_input2_float("transparencyCutoff");
        ommj["binaryShadowTestDirectRay"] = get_input2_bool("binaryShadowTestDirectRay");
        ommj["binaryShadowTestIndirectRay"] = get_input2_bool("binaryShadowTestIndirectRay");

        auto mtl = dynamic_cast<MaterialObject*>(get_input("mtl"));
        auto json = nlohmann::json::parse(mtl->parameters);
        json["omm"] = std::move(ommj);
        mtl->parameters = json.dump();
        
        set_output("mtl", std::move(mtl->clone()));
    }
};

ZENDEFNODE(ConfigOpacityMicroMap, {
    {
        {gParamType_Material, "mtl"},
        {gParamType_String, "path", "", zeno::NoSocket, zeno::ReadPathEdit},
        {"enum Auto RGB Max X Y Z W", "alphaMode", "Auto"},
        {gParamType_Float, "opacityCutoff", "0.99"},
        {gParamType_Float, "transparencyCutoff", "0.89"},
        {gParamType_Bool, "binaryShadowTestDirectRay", "0"},
        {gParamType_Bool, "binaryShadowTestIndirectRay", "1"}
    },
    {
        {gParamType_Material, "mtl"},
    },
    {},
    {
        "shader",
    },
});

} // namespace