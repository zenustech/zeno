#pragma once
#include "ClusterMaterialDevice.h"
namespace zeno_cluster_device {
inline bool sameTextureCoordinates(const TextureBinding& a,const TextureBinding& b) {
    if(a.id<0 || b.id<0 || a.width!=b.width || a.height!=b.height || a.flip!=b.flip)return false;
    for(int i=0;i<6;++i)if(a.transform[i]!=b.transform[i])return false;
    return true;
}
}
