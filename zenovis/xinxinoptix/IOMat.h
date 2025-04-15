#pragma once

#include "zxxglslvec.h"

#ifndef uint
#define uint unsigned int
#endif

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

    float diffraction;
    vec3  diffractColor;

    float subsurface;
    vec3  sssColor;
    vec3  sssParam;
    bool sssFxiedRadius;
    float scatterDistance;
    float scatterStep;
    float smoothness;
    float displacement;
    float shadowReceiver;
    float shadowTerminatorOffset;
    float isHair;
    vec3  mask_value;

    vec3 nrm;
    vec3 emission;
};

struct MatInput {
    vec3 pos;
    vec3 nrm;
    vec3 uv;
    vec3 clr;
    vec3 tang;

    float2 _barys;
    inline float3 barys() const {
        return { 1.0f-_barys.x-_barys.y, _barys.x, _barys.y };
    }
    vec3 e1,e2;
    inline float area() const {
        //assert(false && "Empty function area()\n");
        return 0.5*length(cross(e1,e2))+0.000001;
    }
    vec3 els;
    inline float3 eLength() const {
        //assert(false && "Empty function area()\n");
        return make_float3(els.x, els.y, els.z);
    }

    uint instIdx;
    vec3 instPos;
    vec3 instNrm;
    vec3 instUv;
    vec3 instClr;
    vec3 instTang;
    float NoL;
    float LoV;
    
    float rayLength;
    bool isBackFace;
    bool isShadowRay;
    
    vec3 reflectance;
    vec3 N;
    vec3 T;
    vec3 L;
    vec3 V;
    vec3 H;
    vec3 fresnel;
};
