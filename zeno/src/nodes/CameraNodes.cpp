#include <zeno/zeno.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/eulerangle.h>

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include "zeno/funcs/PrimitiveUtils.h"

namespace zeno {

struct MakeCamera : INode {
    virtual void apply() override {
        auto camera = std::make_unique<CameraObject>();

        camera->pos = get_input2<vec3f>("pos");
        camera->up = get_input2<vec3f>("up");
        camera->view = get_input2<vec3f>("view");
        camera->ffar = get_input2<float>("far");
        camera->fnear = get_input2<float>("near");
        camera->fov = get_input2<float>("fov");
        camera->aperture = get_input2<float>("aperture");
        camera->focalPlaneDistance = get_input2<float>("focalPlaneDistance");

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(MakeCamera)({
    {
        {"vec3f", "pos", "0,0,5"},
        {"vec3f", "up", "0,1,0"},
        {"vec3f", "view", "0,0,-1"},
        {"float", "near", "0.01"},
        {"float", "far", "20000"},
        {"float", "fov", "45"},
        {"float", "aperture", "11"},
        {"float", "focalPlaneDistance", "2.0"},
    },
    {
        {"CameraObject", "camera"},
    },
    {
    },
    {"shader"},
});

struct SetPhysicalCamera : INode {
    virtual void apply() override {
        auto camera = get_input("camera");
        auto &ud = camera->userData();
        ud.set2("aperture", get_input2<float>("aperture"));
        ud.set2("shutter_speed", get_input2<float>("shutter_speed"));
        ud.set2("iso", get_input2<float>("iso"));
        ud.set2("aces", get_input2<bool>("aces"));
        ud.set2("exposure", get_input2<bool>("exposure"));

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(SetPhysicalCamera)({
    {
        "camera",
        {"float", "aperture", "2"},
        {"float", "shutter_speed", "0.04"},
        {"float", "iso", "150"},
        {"bool", "aces", "0"},
        {"bool", "exposure", "0"},
    },
    {
            {"CameraObject", "camera"},
    },
    {
    },
    {"shader"},
});

struct TargetCamera : INode {
    virtual void apply() override {
        auto camera = std::make_unique<CameraObject>();

        auto refUp = zeno::normalize(get_input2<vec3f>("refUp"));
        auto pos = get_input2<vec3f>("pos");
        auto target = get_input2<vec3f>("target");
        auto AF = get_input2<bool>("AutoFocus");
        vec3f view = zeno::normalize(target - pos);
        vec3f right = zeno::cross(view, refUp);
        vec3f up = zeno::cross(right, view);

        camera->pos = pos;
        camera->up = up;
        camera->view = view;
        camera->ffar = get_input2<float>("far");
        camera->fnear = get_input2<float>("near");
        camera->fov = get_input2<float>("fov");
        camera->aperture = get_input2<float>("aperture");
        if(AF){
            camera->focalPlaneDistance = zeno::length(target-pos);
        }else{
            camera->focalPlaneDistance = get_input2<float>("focalPlaneDistance");
        }

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(TargetCamera)({
    {
        {"vec3f", "pos", "0,0,5"},
        {"vec3f", "refUp", "0,1,0"},
        {"vec3f", "target", "0,0,0"},
        {"float", "near", "0.01"},
        {"float", "far", "20000"},
        {"float", "fov", "45"},
        {"float", "aperture", "11"},
        {"bool","AutoFocus","false"},
        {"float", "focalPlaneDistance", "2.0"},
    },
    {
        {"CameraObject", "camera"},
    },
    {
    },
    {"shader"},
});

struct MakeLight : INode {
    virtual void apply() override {
        auto light = std::make_unique<LightObject>();
        light->lightDir = normalize(get_input2<vec3f>("lightDir"));
        light->intensity = get_input2<float>("intensity");
        light->shadowTint = get_input2<vec3f>("shadowTint");
        light->lightHight = get_input2<float>("lightHight");
        light->shadowSoftness = get_input2<float>("shadowSoftness");
        light->lightColor = get_input2<vec3f>("lightColor");
        light->lightScale = get_input2<float>("lightScale");
        light->isEnabled = get_input2<bool>("isEnabled");
        set_output("light", std::move(light));
    }
};

ZENO_DEFNODE(MakeLight)({
    {
        {"vec3f", "lightDir", "1,1,0"},
        {"float", "intensity", "10"},
        {"vec3f", "shadowTint", "0.2,0.2,0.2"},
        {"float", "lightHight", "1000.0"},
        {"float", "shadowSoftness", "1.0"},
        {"vec3f", "lightColor", "1,1,1"},
        {"float", "lightScale", "1"},
        {"bool", "isEnabled", "1"},
    },
    {
        {"LightObject", "light"},
    },
    {
    },
    {"shader"},
});

struct LightNode : INode {
    virtual void apply() override {
        auto isL = true; //get_input2<int>("islight");
        auto invertdir = get_input2<int>("invertdir");
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto quaternion = get_input2<zeno::vec4f>("quaternion");

        auto color = get_input2<zeno::vec3f>("color");

        auto exposure = get_input2<float>("exposure");
        auto intensity = get_input2<float>("intensity");

        auto scaler = powf(2.0f, exposure);

        if (std::isnan(scaler) || std::isinf(scaler) || scaler < 0.0f) {
            scaler = 1.0f;
            printf("Light exposure = %f is invalid, fallback to 0.0 \n", exposure);
        }

        intensity *= scaler;

        std::string type = get_input2<std::string>(lightTypeKey);
        auto typeEnum = magic_enum::enum_cast<LightType>(type).value_or(LightType::Diffuse);
        auto typeOrder = magic_enum::enum_integer(typeEnum);

        std::string shapeString = get_input2<std::string>(lightShapeKey);
        auto shapeEnum = magic_enum::enum_cast<LightShape>(shapeString).value_or(LightShape::Plane);
        auto shapeOrder = magic_enum::enum_integer(shapeEnum);

        auto prim = std::make_shared<zeno::PrimitiveObject>();

        if (has_input("prim")) {
            auto mesh = get_input<PrimitiveObject>("prim");

            if (mesh->size() > 0) {
                prim = mesh;
                shapeEnum = LightShape::TriangleMesh;
                shapeOrder = magic_enum::enum_integer(shapeEnum);
            }
        } else {

        auto &verts = prim->verts;
        auto &tris = prim->tris;

            auto start_point = zeno::vec3f(0.5, 0, 0.5);
            float rm = 1.0f;
            float cm = 1.0f;

            auto order = get_input2<std::string>("EulerRotationOrder:");
            auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

            auto measure = get_input2<std::string>("EulerAngleMeasure:");
            auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

            glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
            glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

            // Plane Verts
            for(int i=0; i<=1; i++){

                auto rp = start_point - zeno::vec3f(i*rm, 0, 0);
                for(int j=0; j<=1; j++){
                    auto p = rp - zeno::vec3f(0, 0, j*cm);
                    // S R Q T
                    p = p * scale;  // Scale
                    auto gp = glm::vec3(p[0], p[1], p[2]);
                    glm::vec4 result = rotation * glm::vec4(gp, 1.0f);  // Rotate
                    gp = glm::vec3(result.x, result.y, result.z);
                    glm::quat rotation(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
                    gp = glm::rotate(rotation, gp);
                    p = zeno::vec3f(gp.x, gp.y, gp.z);
                    auto zp = zeno::vec3f(p[0], p[1], p[2]);
                    zp = zp + position;  // Translate

                    verts.push_back(zp);
                }
            }

            // Plane Indices
            tris.emplace_back(zeno::vec3i(0, 3, 1));
            tris.emplace_back(zeno::vec3i(3, 0, 2));
        }

        auto &verts = prim->verts;
        auto &tris = prim->tris;

        auto &clr = prim->verts.add_attr<zeno::vec3f>("clr");
        auto c = color * intensity;

        for (size_t i=0; i<c.size(); ++i) {
            if (std::isnan(c[i]) || std::isinf(c[i]) || c[i] < 0.0f) {
                c[i] = 1.0f;
                printf("Light color component %llu is invalid, fallback to 1.0 \n", i);
            }
        }

        for(int i=0; i<verts.size(); i++){
            clr[i] = c;
        }

        prim->userData().set2("isRealTimeObject", std::move(isL));

        prim->userData().set2("isL", std::move(isL));
        prim->userData().set2("ivD", std::move(invertdir));
        prim->userData().set2("pos", std::move(position));
        prim->userData().set2("scale", std::move(scale));
        prim->userData().set2("rotate", std::move(rotate));
        prim->userData().set2("quaternion", std::move(quaternion));
        prim->userData().set2("color", std::move(color));
        prim->userData().set2("intensity", std::move(intensity));

        auto fluxFixed = get_input2<float>("fluxFixed");
        prim->userData().set2("fluxFixed", std::move(fluxFixed));
        auto maxDistance = get_input2<float>("maxDistance");
        prim->userData().set2("maxDistance", std::move(maxDistance));
        auto falloffExponent = get_input2<float>("falloffExponent");
        prim->userData().set2("falloffExponent", std::move(falloffExponent));

        auto mask = get_input2<int>("mask");
        auto spread = get_input2<zeno::vec2f>("spread");
        auto visible = get_input2<int>("visible");
        auto doubleside = get_input2<int>("doubleside");

        if (has_input2<std::string>("profile")) {
            auto profile = get_input2<std::string>("profile");
            prim->userData().set2("lightProfile", std::move(profile));
        }
        if (has_input2<std::string>("texturePath")) {
            auto texture = get_input2<std::string>("texturePath");
            prim->userData().set2("lightTexture", std::move(texture));

            auto gamma = get_input2<float>("textureGamma");
            prim->userData().set2("lightGamma", std::move(gamma));
        }

        prim->userData().set2("type", std::move(typeOrder));
        prim->userData().set2("shape", std::move(shapeOrder));

        prim->userData().set2("mask", std::move(mask));
        prim->userData().set2("spread", std::move(spread));
        prim->userData().set2("visible", std::move(visible));
        prim->userData().set2("doubleside", std::move(doubleside));

        auto visibleIntensity = get_input2<float>("visibleIntensity");
        prim->userData().set2("visibleIntensity", std::move(visibleIntensity));

        set_output("prim", std::move(prim));
    }

    const static inline std::string lightShapeKey = "shape";

    static std::string lightShapeDefaultString() {
        auto name = magic_enum::enum_name(LightShape::Plane);
        return std::string(name);
    }

    static std::string lightShapeListString() {
        auto list = magic_enum::enum_names<LightShape>();

        std::string result;
        for (auto& ele : list) {
            result += " ";
            result += ele;
        }
        return result;
    }

    const static inline std::string lightTypeKey = "type";

    static std::string lightTypeDefaultString() {
        auto name = magic_enum::enum_name(LightType::Diffuse);
        return std::string(name);
    }

    static std::string lightTypeListString() {
        auto list = magic_enum::enum_names<LightType>();

        std::string result;
        for (auto& ele : list) {
            result += " ";
            result += ele;
        }
        return result;
    }
};

ZENO_DEFNODE(LightNode)({
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vec4f", "quaternion", "1, 0, 0, 0"},

        {"vec3f", "color", "1, 1, 1"},
        {"float", "exposure", "0"},
        {"float", "intensity", "1"},
        {"float", "fluxFixed", "-1.0"},

        {"vec2f", "spread", "1.0, 0.0"},
        {"float", "maxDistance", "-1.0" },
        {"float", "falloffExponent", "2.0"},
        {"int", "mask", "255"},
        {"bool", "visible", "0"},
        {"bool", "invertdir", "0"},
        {"bool", "doubleside", "0"},

        {"readpath", "profile"},
        {"readpath", "texturePath"},
        {"float",  "textureGamma", "1.0"},
        {"float", "visibleIntensity", "-1.0"},

        {"enum " + LightNode::lightShapeListString(), LightNode::lightShapeKey, LightNode::lightShapeDefaultString()},
        {"enum " + LightNode::lightTypeListString(), LightNode::lightTypeKey, LightNode::lightTypeDefaultString()},
        {"PrimitiveObject", "prim"},
    },
    {
        "prim"
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", EulerAngle::RotationOrderDefaultString()},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", EulerAngle::MeasureDefaultString()}
    },
    {"shader"},
});


struct ScreenSpaceProjectedGrid : INode {
    float hitOnFloor(vec3f pos, vec3f dir, float sea_level) const {
        float t = (sea_level - pos[1]) / dir[1];
        return t;
    }
    virtual void apply() override {
        auto cam = get_input2<CameraObject>("cam");
        auto prim = std::make_unique<PrimitiveObject>();
        auto raw_width = get_input2<int>("width");
        auto raw_height = get_input2<int>("height");
        auto u_padding = get_input2<int>("u_padding");
        auto v_padding = get_input2<int>("v_padding");
        auto sea_level = get_input2<float>("sea_level");
        auto fov = glm::radians(cam->fov);
        auto pos = cam->pos;
        auto up = cam->up;
        auto view = cam->view;
        auto infinite = cam->ffar;

        auto width = raw_width + u_padding * 2;
        auto height = raw_height + v_padding * 2;

        auto right = zeno::cross(view, up);
        float ratio = float(raw_width) / float(raw_height);
        float right_scale = std::tan(fov / 2) * ratio * float(width - 1) / float(raw_width - 1);
        float up_scale = std::tan(fov / 2) * float(height - 1) / float(raw_height - 1);
        prim->verts.resize(width * height);
        for (auto j = 0; j <= height - 1; j++) {
            float v = float(j) / float(height - 1) * 2.0f - 1.0f;
            for (auto i = 0; i <= width - 1; i++) {
                float u = float(i) / float(width - 1) * 2.0f - 1.0f;
                auto dir = view + u * right * right_scale + v * up * up_scale;
                auto ndir = zeno::normalize(dir);
                auto t = hitOnFloor(pos, ndir, sea_level);
                auto index = j * width + i;
                auto &flag = prim->verts.add_attr<int>("flag");
                if (t > 0 && t * zeno::dot(ndir, dir) < infinite) {
                    prim->verts[index] = pos + ndir * t;
                    prim->verts[index][1] = sea_level;
                    flag[index] = 1;
                }
                else {
                    flag[index] = 0;
                }
            }
        }
        std::vector<vec3i> tris;
        tris.reserve((width - 1) * (height - 1) * 2);
        for (auto j = 0; j < height - 1; j++) {
            for (auto i = 0; i < width - 1; i++) {
                auto _0 = j * width + i;
                auto _1 = j * width + i + 1;
                auto _2 = j * width + i + 1 + width;
                auto _3 = j * width + i + width;
                tris.emplace_back(_0, _1, _2);
                tris.emplace_back(_0, _2, _3);
            }
        }
        prim->tris.values = tris;

        primFilterVerts(prim.get(), "flag", 0, true, "", "verts");

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(ScreenSpaceProjectedGrid)({
     {
         "cam",
         {"int", "width", "1920"},
         {"int", "height", "1080"},
         {"int", "u_padding", "0"},
         {"int", "v_padding", "0"},
         {"float", "sea_level", "0"},
     },
     {
         "prim",
     },
     {
     },
     {"shader"},
 });


struct CameraFrustum : INode {
    virtual void apply() override {
        auto cam = get_input2<CameraObject>("cam");
        auto width = get_input2<int>("width");
        auto height = get_input2<int>("height");
        auto fov = glm::radians(cam->fov);
        auto pos = cam->pos;
        auto up = cam->up;
        auto view = cam->view;
        auto fnear = cam->fnear;
        auto ffar = cam->ffar;
        auto right = zeno::cross(view, up);
        float ratio = float(width) / float(height);
        auto prim = std::make_unique<PrimitiveObject>();
        prim->verts.resize(8);
        vec3f _near_left_up = pos + fnear * (view - right * std::tan(fov / 2) * ratio + up * std::tan(fov / 2));
        vec3f _near_left_down = pos + fnear * (view - right * std::tan(fov / 2) * ratio - up * std::tan(fov / 2));
        vec3f _near_right_up = pos + fnear * (view + right * std::tan(fov / 2) * ratio + up * std::tan(fov / 2));
        vec3f _near_right_down = pos + fnear * (view + right * std::tan(fov / 2) * ratio - up * std::tan(fov / 2));
        vec3f _far_left_up = pos + ffar * (view - right * std::tan(fov / 2) * ratio + up * std::tan(fov / 2));
        vec3f _far_left_down = pos + ffar * (view - right * std::tan(fov / 2) * ratio - up * std::tan(fov / 2));
        vec3f _far_right_up = pos + ffar * (view + right * std::tan(fov / 2) * ratio + up * std::tan(fov / 2));
        vec3f _far_right_down = pos + ffar * (view + right * std::tan(fov / 2) * ratio - up * std::tan(fov / 2));
        prim->verts[0] = _near_left_up;
        prim->verts[1] = _near_left_down;
        prim->verts[2] = _near_right_up;
        prim->verts[3] = _near_right_down;
        prim->verts[4] = _far_left_up;
        prim->verts[5] = _far_left_down;
        prim->verts[6] = _far_right_up;
        prim->verts[7] = _far_right_down;

        prim->lines.resize(12);
        prim->lines[0] = {0, 1};
        prim->lines[1] = {2, 3};
        prim->lines[2] = {0, 2};
        prim->lines[3] = {1, 3};
        prim->lines[0 + 4] = vec2i(0, 1) + 4;
        prim->lines[1 + 4] = vec2i(2, 3) + 4;
        prim->lines[2 + 4] = vec2i(0, 2) + 4;
        prim->lines[3 + 4] = vec2i(1, 3) + 4;
        prim->lines[0 + 8] = vec2i(0, 4);
        prim->lines[1 + 8] = vec2i(1, 5);
        prim->lines[2 + 8] = vec2i(2, 6);
        prim->lines[3 + 8] = vec2i(3, 7);

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(CameraFrustum)({
     {
         "cam",
         {"int", "width", "1920"},
         {"int", "height", "1080"},
     },
     {
         "prim",
     },
     {
     },
     {"shader"},
 });

};
