//
// Created by zhouhang on 2023/1/16.
//
#include <zeno/zeno.h>

struct LightPointData {
    zeno::vec3f color;
    float intensity;
    float radius;
    zeno::vec3f translation;
};

struct LightAreaData {
    zeno::vec3f color;
    float intensity;
    zeno::vec3f translation;
    zeno::vec3f eulerXYZ;
    zeno::vec4f quatRotation;
    zeno::vec3f scaling;
};

struct LightSunData {
    zeno::vec3f color;
    float intensity;
    float softness;
    zeno::vec3f translation;
    zeno::vec2f dirUV;
};

struct LightEnvData {
    std::string lightType;
    std::string path;
    zeno::vec3f color;
    float intensity;
    zeno::vec3f translation;
    zeno::vec3f eulerXYZ;
    zeno::vec4f quatRotation;
    zeno::vec3f windDir;
    float timeStart;
    float timeSpeed;
};

struct NewCameraData {
    zeno::vec3f translation;
    zeno::vec3f eulerXYZ;
    zeno::vec4f quatRotation;
    float near;
    float far;
    float fov;
    float aperture;
    float focalPlaneDistance;
    bool useTarget;
    zeno::vec3f target;
};

struct LightPoint : zeno::INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(LightPoint, {
    {
        {"vec3f", "color", "1, 1, 1"},
        {"float", "intensity", "1"},
        {"float", "radius", "0.1"},
        {"vec3f", "translation", "0,0,0"},
    },
    {
    },
    {},
    {"light"},
});


struct LightArea : zeno::INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(LightArea, {
    {
        {"vec3f", "color", "1, 1, 1"},
        {"float", "intensity", "1"},
        {"vec3f", "translation", "0,0,0"},
        {"vec3f", "eulerXYZ", "0,0,0"},
        {"vec4f", "quatRotation", "0,0,0,1"},
        {"vec3f", "scaling", "1,1,1"},
    },
    {
    },
    {},
    {"light"},
});


struct LightSun : zeno::INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(LightSun, {
    {
        {"vec3f", "color", "1, 1, 1"},
        {"float", "intensity", "1"},
        {"float", "temperatureMix", "0"},
        {"float", "temperature", "6500"},
        {"float", "softness", "1"},
        {"vec2f", "dirUV", "-60,45"},
    },
    {
    },
    {},
    {"light"},
});

struct LightEnv : zeno::INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(LightEnv, {
    {
        {"bool", "procedural", "1"},
        {"readpath", "path"},
        {"float", "intensity", "1"},
        {"vec3f", "eulerXYZ", "0,0,0"},
        {"vec4f", "quatRotation", "0,0,0,1"},
        {"vec2f", "windDir", "0,0"},
        {"float", "timeStart", "0"},
        {"float", "timeSpeed", "0.1"},
    },
    {
    },
    {},
    {"light"},
});

struct NewCamera : zeno::INode {
    virtual void apply() override {
    }
};

ZENDEFNODE(NewCamera, {
     {
         {"vec3f", "translation", "0,0,0"},
         {"vec3f", "eulerXYZ", "0,0,0"},
         {"vec4f", "quatRotation", "0,0,0,1"},
         {"float", "near", "0.01"},
         {"float", "far", "20000"},
         {"float", "fov", "45"},
         {"float", "aperture", "0.1"},
         {"float", "focalPlaneDistance", "2.0"},
         {"bool", "useTarget", "0"},
         {"vec3f", "target", "0,0,0"},
     },
     {
     },
     {
     },
     {"light"},
 });
