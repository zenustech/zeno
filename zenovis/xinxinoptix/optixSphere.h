#pragma once 

#include "optixCommon.h"
#include "XAS.h"

#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <zeno/utils/vec.h>

struct SphereTransformed {
    bool dirty = true;
    
    std::string materialID;
    std::string instanceID;
    
    std::shared_ptr<SceneNode> node;
};

struct SphereGroup {
    bool dirty = true;

    std::vector<zeno::vec3f> centerV;
    std::vector<float> radiusV;
    std::vector<zeno::vec3f> colorV;

    std::shared_ptr<SceneNode> node;

    xinxinoptix::raii<CUdeviceptr> color_buffer;
};

void buildUnitSphereGAS(const OptixDeviceContext& context,  OptixTraversableHandle& gas_handle, xinxinoptix::raii<CUdeviceptr>& d_gas_output_buffer);

void buildSphereGroupGAS(const OptixDeviceContext &context, SphereGroup& group);