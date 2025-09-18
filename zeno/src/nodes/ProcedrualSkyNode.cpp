#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/fileio.h>
#include <zeno/ListObject.h>
#include <tinygltf/json.hpp>

#include <zeno/utils/eulerangle.h>
#include <zeno/types/LightObject.h>


namespace zeno {
struct ProceduralSky : INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        prim->userData().set2("isRealTimeObject", std::move(1));
        prim->userData().set2("ProceduralSky", std::move(1));
        prim->userData().set2("sunLightDir", std::move(get_input2<zeno::vec2f>("sunLightDir")));
        prim->userData().set2("sunLightSoftness", std::move(get_input2<float>("sunLightSoftness")));
        prim->userData().set2("windDir", std::move(get_input2<zeno::vec2f>("windDir")));
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
             std::string native_path = std::filesystem::u8path(path).string();
             if (!path.empty() && !file_exists(native_path)) {
                 throw zeno::makeError("HDRSky file not exists");
             }
        }
        prim->userData().set2("isRealTimeObject", std::move(1));
        prim->userData().set2("HDRSky", std::move(path));
        prim->userData().set2("evnTexRotation", std::move(get_input2<float>("rotation")));
        prim->userData().set2("evnTex3DRotation", std::move(get_input2<zeno::vec3f>("rotation3d")));
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
    {"deprecated"},
});

struct HDRSky2 : INode {
    void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::string path = "";
        if (has_input2<std::string>("path")) {
             path = get_input2<std::string>("path");
             std::string native_path = std::filesystem::u8path(path).string();
             if (!path.empty() && !file_exists(native_path)) {
                 throw zeno::makeError("HDRSky file not exists");
             }
        }
        prim->userData().set2("isRealTimeObject", 1);
        prim->userData().set2("HDRSky", path);
        prim->userData().set2("evnTexRotation", 0);
        prim->userData().set2("evnTex3DRotation", get_input2<zeno::vec3f>("rotation3d"));
        prim->userData().set2("evnTexStrength", get_input2<float>("strength"));
        prim->userData().set2("enable", 1);
        set_output("HDRSky", prim);
    }
};

ZENDEFNODE(HDRSky2, {
    {
        {"readpath", "path"},
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

struct DistantLightWrapper : IObject{
    DistantLightData data;
};

struct DistantLight : INode {

    virtual void apply() override {
        auto dir2 = get_input2<zeno::vec2f>("Lat-Lon");
        // dir2[0] = fmod(dir2[0], 180.f);
        // dir2[1] = fmod(dir2[1], 180.f);

        dir2[0] = glm::radians(dir2[0]);
        dir2[1] = glm::radians(dir2[1]);

        zeno::vec3f dir3;
        dir3[1] = std::sin(dir2[0]);
        
        dir3[2] = std::cos(dir2[0]) * std::cos(dir2[1]);
        dir3[0] = std::cos(dir2[0]) * std::sin(dir2[1]);
        auto dir = get_input2<zeno::vec3f>("dir");
        dir3 = length(dir)>0?dir:dir3;
        dir3 = zeno::normalize(dir3);
    
        auto angleExtent = get_input2<float>("angleExtent");
        angleExtent = zeno::clamp(angleExtent, 0.0f, 60.0f);

        auto color = get_input2<zeno::vec3f>("color");
        auto intensity = get_input2<float>("intensity");
        intensity = fmaxf(0.0, intensity);

        auto result = std::make_shared<DistantLightWrapper>();
        result->data.direction = dir3;
        result->data.angle = angleExtent;
        result->data.color = color;
        result->data.intensity = intensity;
        set_output2("out", std::move(result) );
        set_output2("dir", std::move(dir3) );
    }
};

ZENDEFNODE(DistantLight, {
    {
        {"vec2f", "Lat-Lon", "45, 90"},
        {"vec3f", "dir", "0,0,0"},
        {"float", "angleExtent", "0.5"},
        {"colorvec3f", "color", "1,1,1"},
        {"float", "intensity", "1"}
    },
    {
        {"vec3f", "dir"},
        {"out"},
    },
    {
    },
    {"shader"},
});

struct PortalLight : INode {
    virtual void apply() override {

        auto pos = get_input2<zeno::vec3f>("pos");
        auto scale = get_input2<zeno::vec2f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto size = get_input2<int>("size");
        size = std::max(size, 180);

        scale = 0.5f * abs(scale);

        auto order = get_input2<std::string>("EulerRotationOrder:");
        auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::XYZ);

        auto measure = get_input2<std::string>("EulerAngleMeasure:");
        auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

        glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
        glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        glm::mat4 transform(1.0f);

        transform = glm::translate(transform, glm::vec3(pos[0], pos[1], pos[2]));
        transform = transform * rotation;
        transform = glm::scale(transform, glm::vec3(scale[0], 0.5 * (scale[0] + scale[1]), scale[1]));

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->verts->resize(8);

        prim->verts[0] = zeno::vec3f(-1, 0, -1);
        prim->verts[1] = zeno::vec3f(+1, 0, -1);
        prim->verts[2] = zeno::vec3f(+1, 0, +1);
        prim->verts[3] = zeno::vec3f(-1, 0, +1);

        prim->verts[4] = zeno::vec3f(0, 0, 0);
        prim->verts[5] = zeno::vec3f(0.5, 0, 0);
        prim->verts[6] = zeno::vec3f(0, 0.5, 0);
        prim->verts[7] = zeno::vec3f(0, 0, 0.5);

        for (size_t i=0; i<prim->verts->size(); ++i) {
            auto& ele = prim->verts[i];
            auto ttt = transform * glm::vec4(ele[0], ele[1], ele[2], 1.0f);
            prim->verts[i] = zeno::vec3f(ttt.x, ttt.y, ttt.z);
        }

        //prim->lines.attrs.clear();
        prim->lines->resize(8);
        prim->lines[0] = {0, 1};
        prim->lines[1] = {1, 2};
        prim->lines[2] = {2, 3};
        prim->lines[3] = {3, 0};

        prim->lines[4] = {4, 5};
        prim->lines[5] = {4, 6};
        prim->lines[6] = {4, 7};

        auto& color = prim->verts.add_attr<zeno::vec3f>("clr");
        color.resize(8);
        color[0] = {1,1,1};
        color[1] = {1,1,1};
        color[2] = {1,1,1};
        color[3] = {1,1,1};
        
        color[4] = {1, 1, 1};
        color[5] = {1, 0, 0};
        color[6] = {0, 1, 0};
        color[7] = {0, 0, 1};
        //prim->lines.update();
        prim->userData().set2("size", size);
        set_output2("out", std::move(prim));
    }
};

ZENDEFNODE(PortalLight, {
    {
        {"vec3f", "pos", "0,0,0"},
        {"vec2f", "scale", "1, 1"},
        {"vec3f", "rotate", "0,0,0"},
        {"int", "size", "180"}
    },
    {
        {"out"},
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "XYZ"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
    {"shader"},
});

struct SkyComposer : INode {
    virtual void apply() override {

        auto prim = std::make_shared<zeno::PrimitiveObject>();

        if (has_input("dlights")) {
            auto dlights = get_input<ListObject>("dlights")->get<DistantLightWrapper>();
            if (dlights.empty()) {
                throw zeno::makeError("Bad input for dlights");
            }

            prim->verts->resize(dlights.size());
            auto& attr_rad = prim->verts.add_attr<float>("rad");
            auto& attr_angle = prim->verts.add_attr<float>("angle");
            auto& attr_color = prim->verts.add_attr<zeno::vec3f>("color");
            auto& attr_inten = prim->verts.add_attr<float>("inten");

            unsigned i = 0;
            for (const auto& dlight : dlights) {
                
                prim->verts[i] = dlight->data.direction;
                attr_rad[i] = 0.0f;
                attr_angle[i] = dlight->data.angle;
                attr_color[i] = dlight->data.color;
                attr_inten[i] = dlight->data.intensity;

                ++i;
            }
        }

        if (has_input("portals")) {
            auto portals = get_input<ListObject>("portals")->get<zeno::PrimitiveObject>();
            if (portals.empty()) {
                throw zeno::makeError("Bad input for portals");
            }

            using json = nlohmann::json;
            std::vector<zeno::vec3f> raw(4 * portals.size());
            std::vector<int> psizes(portals.size());

            for (size_t i=0; i<portals.size(); ++i) {
                auto &rect = portals[i];

                auto p0 = rect->verts[0];
                auto p1 = rect->verts[1];
                auto p2 = rect->verts[2];
                auto p3 = rect->verts[3];

                /* p0 --- p1 */
                /* --------- */
                /* p3 --- p2 */

                raw[4 * i + 0] = p0;
                raw[4 * i + 1] = p1;
                raw[4 * i + 2] = p2;
                raw[4 * i + 3] = p3;

                auto psize = rect->userData().get2<int>("size");
                psizes[i] = psize;
            }

            json aux(raw);
            prim->userData().set2("portals", std::move(aux.dump()));
            prim->userData().set2("psizes", json(psizes).dump());
        }

        prim->userData().set2("SkyComposer", std::move(1));
        prim->userData().set2("isRealTimeObject", std::move(1));
        set_output2("out", std::move(prim));
    }
};

ZENDEFNODE(SkyComposer, {
    {

        {"list", "dlights"},
        {"list", "portals"}
    },
    {
        {"out"},
    },
    {
        {"enum SphereUnbounded", "proxy", "SphereUnbounded"},
    },
    {"shader"},
});

vec3f colorTemperatureToRGB(float temperatureInKelvins)
{
    vec3f retColor;

    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0f, 40000.0f) / 100.0f;

    if (temperatureInKelvins <= 66.0f)
    {
        retColor[0] = 1.0f;
        retColor[1] = zeno::clamp(0.39008157876901960784f * log(temperatureInKelvins) - 0.63184144378862745098f, 0.0f, 1.0f);
    }
    else
    {
        float t = temperatureInKelvins - 60.0f;
        retColor[0] = zeno::clamp(1.29293618606274509804f * pow(t, -0.1332047592f), 0.0f, 1.0f);
        retColor[1] = zeno::clamp(1.12989086089529411765f * pow(t, -0.0755148492f), 0.0f, 1.0f);
    }

    if (temperatureInKelvins >= 66.0f)
        retColor[2] = 1.0;
    else if(temperatureInKelvins <= 19.0f)
        retColor[2] = 0.0;
    else
        retColor[2] = zeno::clamp(0.54320678911019607843f * log(temperatureInKelvins - 10.0f) - 1.19625408914f, 0.0f, 1.0f);

    return retColor;
}

struct Blackbody : INode {
    virtual void apply() override {
        float temperature = get_input2<float>("temperature");
        temperature = zeno::clamp(temperature, 1000.0f, 40000.0f);
        auto color = colorTemperatureToRGB(temperature);
        set_output2("color", color);
    }
};

ZENDEFNODE(Blackbody, {
    {
        {"float", "temperature", "6500"},
    },
    {
        {"color"},
    },
    {
    },
    {"shader"},
});
};
