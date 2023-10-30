#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/fileio.h>

namespace zeno {
struct ProceduralSky : INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        prim->userData().set2("isRealTimeObject", std::move(1));
        prim->userData().set2("ProceduralSky", std::move(1));
        prim->userData().set2("sunLightDir", std::move(get_input2<vec2f>("sunLightDir")));
        prim->userData().set2("sunLightSoftness", std::move(get_input2<float>("sunLightSoftness")));
        prim->userData().set2("windDir", std::move(get_input2<vec2f>("windDir")));
        prim->userData().set2("timeStart", std::move(get_input2<float>("timeStart")));
        prim->userData().set2("timeSpeed", std::move(get_input2<float>("timeSpeed")));
        prim->userData().set2("sunLightIntensity", std::move(get_input2<float>("sunLightIntensity")));
        prim->userData().set2("colorTemperatureMix", std::move(get_input2<float>("colorTemperatureMix")));
        prim->userData().set2("colorTemperature", std::move(get_input2<float>("colorTemperature")));
        set_output("ProceduralSky", std::move(prim));
    }
};

ZENDEFNODE(ProceduralSky, {
        {
                {"vec2f", "sunLightDir", "-60,45"},
                {"float", "sunLightSoftness", "1"},
                {"float", "sunLightIntensity", "1"},
                {"float", "colorTemperatureMix", "0"},
                {"float", "colorTemperature", "6500"},
                {"vec2f", "windDir", "0,0"},
                {"float", "timeStart", "0"},
                {"float", "timeSpeed", "0.1"},
        },
        {
                {"ProceduralSky"},
        },
        {
        },
        {"shader"},
});

struct HDRSky : INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::string path = "";
        if (has_input2<std::string>("path")) {
             path = get_input2<std::string>("path");
             if (!file_exists(path)) {
                 throw zeno::makeError("HDRSky file not exists");
             }
        }
        prim->userData().set2("isRealTimeObject", std::move(1));
        prim->userData().set2("HDRSky", std::move(path));
        prim->userData().set2("evnTexRotation", std::move(get_input2<float>("rotation")));
        prim->userData().set2("evnTex3DRotation", std::move(get_input2<vec3f>("rotation3d")));
        prim->userData().set2("evnTexStrength", std::move(get_input2<float>("strength")));
        prim->userData().set2("enable", std::move(get_input2<bool>("enable")));
        set_output("HDRSky", std::move(prim));
    }
};

ZENDEFNODE(HDRSky, {
    {
        {"bool", "enable", "1"},
        {"readpath", "path"},
        {"float", "rotation", "0"},
        {"vec3f", "rotation3d", "0,0,0"},
        {"float", "strength", "1"},
    },
    {
        {"HDRSky"},
    },
    {
    },
    {"shader"},
});
};
