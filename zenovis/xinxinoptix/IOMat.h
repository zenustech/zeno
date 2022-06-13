#pragma once

#include "zxxglslvec.h"

struct MatOutput {
    vec3  basecolor;
    float metallic;
    float roughness;
    float subsurface;
    float specular;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float opacity;
    vec3 nrm;
    vec3 emission;
};

struct MatInput {
    vec3 pos;
    vec3 nrm;
    vec3 uv;
    vec3 clr;
    vec3 tang;
};
