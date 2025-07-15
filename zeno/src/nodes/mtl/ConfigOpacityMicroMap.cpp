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

        auto path = get_input2<std::string>("path", "");
        bool exist = std::filesystem::exists(path);
        if (!exist) {
            throw std::runtime_error("File " + path + " doesn't exist");
        }
        nlohmann::json ommj;
        ommj["path"] = path;
        ommj["alphaMode"] = get_input2<std::string>("alphaMode");
        ommj["opacityCutoff"] = get_input2<float>("opacityCutoff");
        ommj["transparencyCutoff"] = get_input2<float>("transparencyCutoff");
        ommj["binaryShadowTestDirectRay"] = get_input2<bool>("binaryShadowTestDirectRay");
        ommj["binaryShadowTestIndirectRay"] = get_input2<bool>("binaryShadowTestIndirectRay");

        auto mtl = get_input2<MaterialObject>("mtl");
        auto json = nlohmann::json::parse(mtl->parameters);
        json["omm"] = std::move(ommj);
        mtl->parameters = json.dump();
        
        set_output("mtl", std::move(mtl));
    }
};

ZENDEFNODE(ConfigOpacityMicroMap, {
    {
        {"MaterialObject", "mtl"},
        {"readpath", "path"},
        {"enum Auto RGB Max X Y Z W", "alphaMode", "Auto"},
        {"float", "opacityCutoff", "0.99"},
        {"float", "transparencyCutoff", "0.89"},
        {"bool", "binaryShadowTestDirectRay", "0"},
        {"bool", "binaryShadowTestIndirectRay", "1"}
    },
    {
        {"MaterialObject", "mtl"},
    },
    {},
    {
        "shader",
    },
});

} // namespace