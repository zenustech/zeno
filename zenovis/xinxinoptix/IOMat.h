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
    float anisoRotation;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float clearcoatRoughness;
    float clearcoatIOR;
    float opacity;
    float ior;
    float flatness;
    float specTrans;
    float scatterDistance;
    float thin;
    float doubleSide;
    float scatterStep;
    float smoothness;
    vec3  sssColor;
    vec3  sssParam;
    float displacement;
    vec3 reflectance;

    vec3 nrm;
    vec3 emission;
    float vol_anisotropy;
};

struct MatInput {
    vec3 pos;
    vec3 nrm;
    vec3 uv;
    vec3 clr;
    vec3 tang;
    vec3 instPos;
    vec3 instNrm;
    vec3 instUv;
    vec3 instClr;
    vec3 instTang;
    float NoL;
    float LoV;
    vec3 reflectance;
    vec3 N;
    vec3 T;
    vec3 L;
    vec3 V;
    vec3 H;
    vec3 fresnel;
};
