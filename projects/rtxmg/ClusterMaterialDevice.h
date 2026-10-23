#pragma once
#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <vector_types.h>

namespace zeno_cluster_device {
enum MaterialFlags { AlphaMask=1, AlphaBlend=2, DoubleSided=4, Unlit=8 };
enum TextureSlot { BaseColor, MetallicRoughness, Normal, Emissive, Specular, SpecularColor, Transmission, Occlusion, TextureSlots };
struct TextureBinding {
    int id=-1;
    unsigned int width=1,height=1;
    unsigned int swizzle=0x03020100u,flip=0;
    float transform[6]={1,0,0,0,1,0};
};
struct Material {
    float baseColor[4]={1,1,1,1},emission[3]={0,0,0},specularColor[3]={1,1,1};
    float metallic=1,roughness=1,normalScale=1,specular=1,ior=1.5f,transmission=0,alphaCutoff=.5f,occlusionStrength=1;
    unsigned int flags=0;
    TextureBinding textures[TextureSlots];
};
struct PlacementMaterials { unsigned int offset,count; float transform[12]; };
struct CoarseTexture { unsigned long long texture; unsigned int layer,reserved; };
struct MaterialLaunch {
    // Launch parameters must remain POD: OptiX declares Params __constant__.
    const Material* materials;
    const PlacementMaterials* placements;
    const unsigned int* materialIds;
    unsigned int materialCount,placementCount;
    unsigned int* missing;
    unsigned int diagnostics;
    unsigned int twoSidedOpaque;
    const CoarseTexture* coarseTextures;
    demandLoading::DeviceContext textures;
};
}
