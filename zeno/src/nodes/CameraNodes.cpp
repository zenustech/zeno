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
#include "zeno/extra/TempNode.h"
#include <regex>

namespace zeno {
struct CameraNode: zeno::INode{
    virtual void apply() override {
        auto camera = std::make_unique<zeno::CameraObject>();

        camera->pos = get_input2<zeno::vec3f>("pos");
        camera->up = get_input2<zeno::vec3f>("up");
        camera->view = get_input2<zeno::vec3f>("view");
        camera->fov = get_input2<float>("fov");
        camera->aperture = get_input2<float>("aperture");
        camera->focalPlaneDistance = get_input2<float>("focalPlaneDistance");
        camera->userData().set2("frame", get_input2<float>("frame"));

        auto other_props = get_input2<std::string>("other");
        std::regex reg(",");
        std::sregex_token_iterator p(other_props.begin(), other_props.end(), reg, -1);
        std::sregex_token_iterator end;
        std::vector<float> prop_vals;
        while (p != end) {
            prop_vals.push_back(std::stof(*p));
            p++;
        }
        if (prop_vals.size() == 6) {
            camera->pivot = {prop_vals[0], prop_vals[1], prop_vals[2]};
        }

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(CameraNode)({
     {
         {"vec3f", "pos", "0,0,5"},
         {"vec3f", "up", "0,1,0"},
         {"vec3f", "view", "0,0,-1"},
         {"float", "fov", "45"},
         {"float", "aperture", "11"},
         {"float", "focalPlaneDistance", "2.0"},
         {"string", "other", ""},
         {"int", "frame", "0"},
     },
     {
         {"CameraObject", "camera"},
     },
     {
     },
     {"FBX"},
 });

struct MakeCamera : INode {
    virtual void apply() override {
        auto camera = std::make_unique<CameraObject>();

        camera->pos = get_input2<zeno::vec3f>("pos");
        camera->up = get_input2<zeno::vec3f>("up");
        camera->view = get_input2<zeno::vec3f>("view");
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
        ud.set2("renderRatio", get_input2<int>("renderRatio"));
        ud.set2("aces", get_input2<bool>("aces"));
        ud.set2("exposure", get_input2<bool>("exposure"));
        ud.set2("panorama_camera", get_input2<bool>("panorama_camera"));
        ud.set2("panorama_vr180", get_input2<bool>("panorama_vr180"));
        ud.set2("pupillary_distance", get_input2<float>("pupillary_distance"));

        set_output("camera", std::move(camera));
    }
};

ZENO_DEFNODE(SetPhysicalCamera)({
    {
        "camera",
        {"float", "aperture", "2"},
        {"float", "shutter_speed", "0.04"},
        {"float", "iso", "150"},
        {"int", "renderRatio", "1"},
        {"bool", "aces", "0"},
        {"bool", "exposure", "0"},
        {"bool", "exposure", "0"},
        {"bool", "panorama_camera", "0"},
        {"bool", "panorama_vr180", "0"},
        {"float", "pupillary_distance", "0.06"},
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

        auto refUp = zeno::normalize(get_input2<zeno::vec3f>("refUp"));
        auto pos = get_input2<zeno::vec3f>("pos");
        auto target = get_input2<zeno::vec3f>("target");
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
        light->lightDir = normalize(get_input2<zeno::vec3f>("lightDir"));
        light->intensity = get_input2<float>("intensity");
        light->shadowTint = get_input2<zeno::vec3f>("shadowTint");
        light->lightHight = get_input2<float>("lightHight");
        light->shadowSoftness = get_input2<float>("shadowSoftness");
        light->lightColor = get_input2<zeno::vec3f>("lightColor");
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

struct ScreenSpaceProjectedGrid : INode {
    float hitOnFloor(vec3f pos, vec3f dir, float sea_level) const {
        float t = (sea_level - pos[1]) / dir[1];
        return t;
    }
    virtual void apply() override {
        auto cam = get_input2<CameraObject>("cam");
        auto prim = std::make_shared<PrimitiveObject>();
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
                if (t > 0 && t * zeno::dot(ndir, dir) < infinite) {
                    prim->verts[j * width + i] = pos + ndir * t;
                }
                else {
                    prim->verts[j * width + i] = pos + dir * infinite;
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

        auto outs = zeno::TempNodeSimpleCaller("PrimitiveClip")
                .set("prim", std::move(prim))
                .set2<vec3f>("origin", pos)
                .set2<vec3f>("direction", view)
                .set2<float>("distance", infinite * 0.999)
                .set2<bool>("reverse:", false)
                .call();

        // Create nodes
        auto new_prim = std::dynamic_pointer_cast<PrimitiveObject>(outs.get("outPrim"));
        for (auto i = 0; i < new_prim->verts.size(); i++) {
            new_prim->verts[i][1] = sea_level;
        }
        set_output("prim", std::move(new_prim));
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
