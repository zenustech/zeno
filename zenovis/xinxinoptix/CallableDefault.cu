#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "IOMat.h"

//COMMON_CODE

extern "C" __device__ MatOutput __direct_callable__evalmat(cudaTextureObject_t zenotex[], float4* uniforms, const MatInput& attrs) {

    /* MODMA */
    auto att_pos = attrs.pos;
    auto att_clr = attrs.clr;
    auto att_uv = attrs.uv;
    auto att_nrm = attrs.nrm;
    auto att_tang = attrs.tang;
    auto att_instPos = attrs.instPos;
    auto att_instNrm = attrs.instNrm;
    auto att_instUv = attrs.instUv;
    auto att_instClr = attrs.instClr;
    auto att_instTang = attrs.instTang;
    auto att_NoL      = attrs.NoL;
    auto att_LoV      = attrs.LoV;
    auto att_N        = attrs.N;
    auto att_T        = attrs.T;
    auto att_L        = attrs.L;
    auto att_V        = attrs.V;
    auto att_H        = attrs.H;
    auto att_reflectance = attrs.reflectance;
    auto att_fresnel  = attrs.fresnel;

#ifndef _FALLBACK_

    /** generated code here beg **/
    //GENERATED_BEGIN_MARK
    /* MODME */
    float mat_base = 1.0f;
    vec3 mat_basecolor = vec3(1.0f, 1.0f, 1.0f);
    float mat_roughness = 0.5f;
    float mat_metallic = 0.0f;
    vec3 mat_metalColor = vec3(1.0f,1.0f,1.0f);
    float mat_specular = 0.0f;
    float mat_specularTint = 0.0f;
    float mat_anisotropic = 0.0f;
    float mat_anisoRotation = 0.0f;

    float mat_subsurface = 0.0f;
    vec3  mat_sssParam = vec3(0.0f,0.0f,0.0f);
    vec3  mat_sssColor = vec3(0.0f,0.0f,0.0f);
    float mat_scatterDistance = 0.0f;
    float mat_scatterStep = 0.0f;
    
    float mat_sheen = 0.0f;
    float mat_sheenTint = 0.0f;

    float mat_clearcoat = 0.0f;
    vec3 mat_clearcoatColor = vec3(1.0f,1.0f,1.0f);
    float mat_clearcoatRoughness = 0.0f;
    float mat_clearcoatIOR = 1.5f;
    float mat_opacity = 0.0f;

    float mat_specTrans = 0.0f;
    vec3 mat_transColor = vec3(1.0f,1.0f,1.0f);
    vec3 mat_transTint = vec3(1.0f,1.0f,1.0f);
    float mat_transTintDepth = 0.0f;
    float mat_transDistance = 0.0f;
    vec3 mat_transScatterColor = vec3(1.0f,1.0f,1.0f);
    float mat_ior = 1.0f;

    float mat_flatness = 0.0f;
    float mat_thin = 0.0f;
    float mat_doubleSide= 0.0f;
    float mat_smoothness = 0.0f;
    vec3  mat_normal = vec3(0.0f, 0.0f, 1.0f);
    float mat_emissionIntensity = float(0);
    vec3 mat_emission = vec3(1.0f, 1.0f, 1.0f);
    float mat_displacement = 0.0f;
    float mat_shadowReceiver = 0.0f;
    float mat_NoL = 1.0f;
    float mat_LoV = 1.0f;
    vec3 mat_reflectance = att_reflectance;
    
    bool sssFxiedRadius = false;

    //GENERATED_END_MARK
    /** generated code here end **/

#else

    float mat_base = 1.0f;
    vec3 mat_basecolor = vec3(1.0f, 1.0f, 1.0f);
    float mat_roughness = 0.5f;
    float mat_metallic = 0.0f;
    vec3 mat_metalColor = vec3(1.0f,1.0f,1.0f);
    float mat_specular = 0.0f;
    float mat_specularTint = 0.0f;
    float mat_anisotropic = 0.0f;
    float mat_anisoRotation = 0.0f;

    float mat_subsurface = 0.0f;
    vec3  mat_sssParam = vec3(0.0f,0.0f,0.0f);
    vec3  mat_sssColor = vec3(0.0f,0.0f,0.0f);
    float mat_scatterDistance = 0.0f;
    float mat_scatterStep = 0.0f;
    
    float mat_sheen = 0.0f;
    float mat_sheenTint = 0.0f;

    float mat_clearcoat = 0.0f;
    vec3 mat_clearcoatColor = vec3(1.0f,1.0f,1.0f);
    float mat_clearcoatRoughness = 0.0f;
    float mat_clearcoatIOR = 1.5f;
    float mat_opacity = 0.0f;

    float mat_specTrans = 0.0f;
    vec3 mat_transColor = vec3(1.0f,1.0f,1.0f);
    vec3 mat_transTint = vec3(1.0f,1.0f,1.0f);
    float mat_transTintDepth = 0.0f;
    float mat_transDistance = 0.0f;
    vec3 mat_transScatterColor = vec3(1.0f,1.0f,1.0f);
    float mat_ior = 1.0f;

    float mat_flatness = 0.0f;
    float mat_thin = 0.0f;
    float mat_doubleSide= 0.0f;
    float mat_smoothness = 0.0f;
    vec3  mat_normal = vec3(0.0f, 0.0f, 1.0f);
    float mat_emissionIntensity = float(0);
    vec3 mat_emission = vec3(1.0f, 1.0f, 1.0f);
    float mat_displacement = 0.0f;
    float mat_shadowReceiver = 0.0f;
    float mat_NoL = 1.0f;
    float mat_LoV = 1.0f;
    vec3 mat_reflectance = att_reflectance;
    
    bool sssFxiedRadius = false;

#endif // _FALLBACK_

    MatOutput mats;
    /* MODME */
    mats.basecolor = mat_base * mat_basecolor;
    mats.roughness = clamp(mat_roughness, 0.01, 0.99);
    mats.metallic = clamp(mat_metallic, 0.0f, 1.0f);
    mats.metalColor = mat_metalColor;
    mats.specular = mat_specular;
    mats.specularTint = mat_specularTint;
    mats.anisotropic = clamp(mat_anisotropic, 0.0f, 1.0f);
    mats.anisoRotation = clamp(mat_anisoRotation, 0.0f, 1.0f);

    mats.subsurface = mat_subsurface;
    mats.sssColor = mat_sssColor;
    mats.sssParam = mat_sssParam;
    mats.scatterDistance = max(0.0f,mat_scatterDistance);
    mats.scatterStep = clamp(mat_scatterStep,0.0f,1.0f);

    mats.sheen = mat_sheen;
    mats.sheenTint = mat_sheenTint;

    mats.clearcoat = clamp(mat_clearcoat, 0.0f, 1.0f);
    mats.clearcoatColor = mat_clearcoatColor;
    mats.clearcoatRoughness = clamp(mat_clearcoatRoughness, 0.01, 0.99);
    mats.clearcoatIOR = mat_clearcoatIOR;
    
    mats.specTrans = clamp(mat_specTrans, 0.0f, 1.0f);
    mats.transColor = mat_transColor;
    mats.transTint = mat_transTint;
    mats.transTintDepth = max(0.0f,mat_transTintDepth);
    mats.transDistance = max(mat_transDistance,0.1f);
    mats.transScatterColor = mat_transScatterColor;
    mats.ior = max(0.0f,mat_ior);

    mats.opacity = mat_opacity;
    mats.nrm = mat_normal;
    mats.emission = mat_emissionIntensity * mat_emission;

    mats.flatness = mat_flatness;
    mats.thin = mat_thin;
    mats.doubleSide = mat_doubleSide;
    mats.shadowReceiver = mat_shadowReceiver;

    mats.sssFxiedRadius = sssFxiedRadius;
    mats.smoothness = mat_smoothness;

    return mats;
}