#pragma once

#include "zxxglslvec.h"

struct MatOutput {
    vec3 basecolor;
    float roughness;
    float opacity;
    float thin;
    float flatness;
    float doubleSide;
    float anisotropic;
    float anisoRotation;
    float ior;
    vec3 reflectance;

    float metallic;
    vec3 metalColor;

    float specular;
    float specularTint;
    float sheen;
    float sheenTint;

    float clearcoat;
    vec3 clearcoatColor;
    float clearcoatRoughness;
    float clearcoatIOR;

    float specTrans;
    vec3 transColor;
    vec3 transTint;
    float transTintDepth;
    float transDistance;
    vec3 transScatterColor;



    float subsurface;
    vec3  sssColor;
    vec3  sssParam;
    float scatterDistance;
    float scatterStep;
    float smoothness;
    float displacement;
    float shadowReceiver;

    vec3 nrm;
    vec3 emission;
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
    float rayLength;
    vec3 reflectance;
    vec3 N;
    vec3 T;
    vec3 L;
    vec3 V;
    vec3 H;
    vec3 fresnel;
};
