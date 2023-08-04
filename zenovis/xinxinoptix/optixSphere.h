#pragma once 

#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Exception.h>

#include <set>
#include <map>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <zeno/utils/vec.h>

#include "XAS.h"
#include "raiicuda.h"

namespace xinxinoptix 
{

struct InfoSphereTransformed {
    std::string materialID;
    std::string instanceID;
    
    glm::mat4 optix_transform;
};

inline std::map<std::string, InfoSphereTransformed> SphereTransformedLUT;
void preload_sphere_transformed(std::string const &key, std::string const &mtlid, const std::string &instID, const glm::mat4& transform);

struct SphereInstanceGroupBase {
    std::string key;
    std::string instanceID;
    std::string materialID;

    zeno::vec3f center{};
    float radius{};
};

inline std::map<std::string, SphereInstanceGroupBase> SpheresInstanceGroupMap;
void preload_sphere_instanced(std::string const &key, std::string const &mtlid, const std::string &instID, const float &radius, const zeno::vec3f &center);

inline std::set<std::string> sphere_unique_mats;
inline std::set<std::string> uniqueMatsForSphere() {
    return sphere_unique_mats;
}

inline void cleanupSpheres() {
    sphere_unique_mats.clear();
    SphereTransformedLUT.clear();
    SpheresInstanceGroupMap.clear();
}

inline OptixTraversableHandle uniformed_sphere_gas_handle {};
inline raii<CUdeviceptr>      uniformed_sphere_gas_buffer {};

void buildUniformedSphereGAS(const OptixDeviceContext& context,  OptixTraversableHandle& gas_handle, raii<CUdeviceptr>& d_gas_output_buffer);

struct SphereInstanceAgent {
    SphereInstanceGroupBase base{};

    std::vector<float> radius_list{};
    std::vector<zeno::vec3f> center_list{};

    raii<CUdeviceptr>      inst_sphere_gas_buffer {};
    OptixTraversableHandle inst_sphere_gas_handle {};

    SphereInstanceAgent(SphereInstanceGroupBase _base):base(_base){}
    //SphereInstanceAgent(SphereInstanceBase &_base):base(_base){}

    ~SphereInstanceAgent() {
        inst_sphere_gas_handle = 0;
        inst_sphere_gas_buffer.reset();
    }
};

inline std::vector<std::shared_ptr<SphereInstanceAgent>> sphereInstanceGroupAgentList;
void buildInstancedSpheresGAS(const OptixDeviceContext &context, std::vector<std::shared_ptr<SphereInstanceAgent>>& agentList);

void updateSphereXAS();

} // NAMESPACE END