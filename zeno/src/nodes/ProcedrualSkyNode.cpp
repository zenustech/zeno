#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/fileio.h>
#include <zeno/types/ListObject_impl.h>
#include <tinygltf/json.hpp>

#include <zeno/utils/eulerangle.h>
#include <zeno/types/LightObject.h>


namespace zeno {
struct ProceduralSky : INode {
    virtual void apply() override {
        auto prim = std::make_unique<zeno::PrimitiveObject>();
        auto pUserData = dynamic_cast<UserData*>(prim->userData());
        pUserData->set2("isRealTimeObject", std::move(1));
        pUserData->set2("ProceduralSky", std::move(1));
        pUserData->set2("sunLightDir", std::move(ZImpl(get_input2<zeno::vec2f>("sunLightDir"))));
        pUserData->set2("sunLightSoftness", std::move(ZImpl(get_input2<float>("sunLightSoftness"))));
        pUserData->set2("windDir", std::move(ZImpl(get_input2<zeno::vec2f>("windDir"))));
        pUserData->set2("timeStart", std::move(ZImpl(get_input2<float>("timeStart"))));
        pUserData->set2("timeSpeed", std::move(ZImpl(get_input2<float>("timeSpeed"))));
        pUserData->set2("sunLightIntensity", std::move(ZImpl(get_input2<float>("sunLightIntensity"))));
        pUserData->set2("colorTemperatureMix", std::move(ZImpl(get_input2<float>("colorTemperatureMix"))));
        pUserData->set2("colorTemperature", std::move(ZImpl(get_input2<float>("colorTemperature"))));
        ZImpl(set_output("ProceduralSky", std::move(prim)));
    }
};

ZENDEFNODE(ProceduralSky, {
        {
                {gParamType_Vec2f, "sunLightDir", "-60,45"},
                {gParamType_Float, "sunLightSoftness", "1"},
                {gParamType_Float, "sunLightIntensity", "1"},
                {gParamType_Float, "colorTemperatureMix", "0"},
                {gParamType_Float, "colorTemperature", "6500"},
                {gParamType_Vec2f, "windDir", "0,0"},
                {gParamType_Float, "timeStart", "0"},
                {gParamType_Float, "timeSpeed", "0.1"},
        },
        {
                {gParamType_Primitive, "ProceduralSky"},
        },
        {
        },
        {"shader"},
});

struct HDRSky : INode {
    virtual void apply() override {
        auto prim = std::make_unique<zeno::PrimitiveObject>();
        std::string path = "";
        if (ZImpl(has_input2<std::string>("path"))) {
             path = ZImpl(get_input2<std::string>("path"));
             std::string native_path = std::filesystem::u8path(path).string();
             if (!path.empty() && !file_exists(native_path)) {
                 throw zeno::makeError("HDRSky file not exists");
             }
        }
        auto pUserData = dynamic_cast<UserData*>(prim->userData());
        pUserData->set2("isRealTimeObject", std::move(1));
        pUserData->set2("HDRSky", std::move(path));
        pUserData->set2("evnTexRotation", std::move(ZImpl(get_input2<float>("rotation"))));
        pUserData->set2("evnTex3DRotation", std::move(ZImpl(get_input2<zeno::vec3f>("rotation3d"))));
        pUserData->set2("evnTexStrength", std::move(ZImpl(get_input2<float>("strength"))));
        pUserData->set2("enable", std::move(ZImpl(get_input2<bool>("enable"))));
        ZImpl(set_output("HDRSky", std::move(prim)));
    }
};

ZENDEFNODE(HDRSky, {
    {
        {gParamType_Bool, "enable", "1"},
        {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
        {gParamType_Float, "rotation", "0"},
        {gParamType_Vec3f, "rotation3d", "0,0,0"},
        {gParamType_Float, "strength", "1"},
    },
    {
        {gParamType_Primitive, "HDRSky"},
    },
    {
    },
    {"shader"},
});

struct DistantLightWrapper : IObject{
    zany clone() const {
        return std::make_unique<DistantLightWrapper>(*this);
    }

    DistantLightData data;
};

struct DistantLight : INode {

    virtual void apply() override {
        auto dir2 = ZImpl(get_input2<zeno::vec2f>("Lat-Lon"));
        // dir2[0] = fmod(dir2[0], 180.f);
        // dir2[1] = fmod(dir2[1], 180.f);

        dir2[0] = glm::radians(dir2[0]);
        dir2[1] = glm::radians(dir2[1]);

        zeno::vec3f dir3;
        dir3[1] = std::sin(dir2[0]);
        
        dir3[2] = std::cos(dir2[0]) * std::cos(dir2[1]);
        dir3[0] = std::cos(dir2[0]) * std::sin(dir2[1]);
        auto dir = ZImpl(get_input2<zeno::vec3f>("dir"));
        dir3 = length(dir)>0?dir:dir3;
        dir3 = zeno::normalize(dir3);
        
        auto angleExtent = ZImpl(get_input2<float>("angleExtent"));
        angleExtent = zeno::clamp(angleExtent, 0.0f, 60.0f);

        auto color = ZImpl(get_input2<zeno::vec3f>("color"));
        auto intensity = ZImpl(get_input2<float>("intensity"));
        intensity = fmaxf(0.0, intensity);

        auto result = std::make_unique<DistantLightWrapper>();
        result->data.direction = dir3;
        result->data.angle = angleExtent;
        result->data.color = color;
        result->data.intensity = intensity;
        ZImpl(set_output("out", std::move(result) ));
        ZImpl(set_output2("dir", std::move(dir3) ));
    }
};

ZENDEFNODE(DistantLight, {
    {
        {gParamType_Vec2f, "Lat-Lon", "45, 90"},
        {gParamType_Vec3f, "dir", "0,0,0"},
        {gParamType_Float, "angleExtent", "0.5"},
        {gParamType_Vec3f, "color", "1,1,1"},
        {gParamType_Float, "intensity", "1"}
    },
    {
        {gParamType_Vec3f, "dir"},
        {gParamType_IObject, "out"},
    },
    {
    },
    {"shader"},
});

struct PortalLight : INode {
    virtual void apply() override {

        auto pos = ZImpl(get_input2<zeno::vec3f>("pos"));
        auto scale = ZImpl(get_input2<zeno::vec2f>("scale"));
        auto rotate = ZImpl(get_input2<zeno::vec3f>("rotate"));
        auto size = ZImpl(get_input2<int>("size"));
        size = std::max(size, 180);

        scale = 0.5f * abs(scale);

        auto order = ZImpl(get_input2<std::string>("EulerRotationOrder:"));
        auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::XYZ);

        auto measure = ZImpl(get_input2<std::string>("EulerAngleMeasure:"));
        auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

        glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
        glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        glm::mat4 transform(1.0f);

        transform = glm::translate(transform, glm::vec3(pos[0], pos[1], pos[2]));
        transform = transform * rotation;
        transform = glm::scale(transform, glm::vec3(scale[0], 0.5 * (scale[0] + scale[1]), scale[1]));

        auto prim = std::make_unique<zeno::PrimitiveObject>();
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
        prim->userData()->set_int("size", size);
        ZImpl(set_output("out", std::move(prim)));
    }
};

ZENDEFNODE(PortalLight, {
    {
        {gParamType_Vec3f, "pos", "0,0,0"},
        {gParamType_Vec2f, "scale", "1, 1"},
        {gParamType_Vec3f, "rotate", "0,0,0"},
        {gParamType_Int, "size", "180"}
    },
    {
        {gParamType_Primitive, "out"},
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "XYZ"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
    {"shader"},
});

struct SkyComposer : INode {
    virtual void apply() override {

        auto prim = std::make_unique<zeno::PrimitiveObject>();

        if (ZImpl(has_input("dlights"))) {
            auto dlights = ZImpl(get_input<ListObject>("dlights"))->m_impl->get<DistantLightWrapper>();
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

        auto pUserData = dynamic_cast<UserData*>(prim->userData());
        if (ZImpl(has_input("portals"))) {
            auto portals = ZImpl(get_input<ListObject>("portals"))->m_impl->get<zeno::PrimitiveObject>();
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

                auto psize = rect->userData()->get_int("size");
                psizes[i] = psize;
            }

            json aux(raw);
            pUserData->set2("portals", std::move(aux.dump()));
            pUserData->set2("psizes", json(psizes).dump());
        }

        pUserData->set2("SkyComposer", std::move(1));
        pUserData->set2("isRealTimeObject", std::move(1));
        ZImpl(set_output("out", std::move(prim)));
    }
};

ZENDEFNODE(SkyComposer, {
    {

        {gParamType_List, "dlights"},
        {gParamType_List, "portals"}
    },
    {
        {gParamType_Primitive, "out"},
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
        float temperature = ZImpl(get_input2<float>("temperature"));
        temperature = zeno::clamp(temperature, 1000.0f, 40000.0f);
        auto color = colorTemperatureToRGB(temperature);
        ZImpl(set_output2("color", color));
    }
};

ZENDEFNODE(Blackbody, {
    {
        {gParamType_Float, "temperature", "6500"},
    },
    {
        {gParamType_Vec3f,"color"},
    },
    {
    },
    {"shader"},
});
};

