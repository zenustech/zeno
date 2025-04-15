#pragma once

#include <set>
#include <map>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tuple>
#include <unordered_map>

#include <tuple>
#include <array>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>

#include <vector_types.h>
#include <zeno/types/NumericObject.h>
#include <tbb/parallel_for.h>

#include <xinxinoptixapi.h>

#include <optix.h>
#include <optix_stubs.h>

#include <raiicuda.h>

#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <tinygltf/json.hpp>

#include "XAS.h"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/fwd.hpp"
#include "optixPathTracer.h"
#include "optixSphere.h"
#include "TypeCaster.h"
#include "optix_types.h"
#include "zeno/utils/vec.h"

#include "optixSphere.h"
#include "optixTriMesh.h"

using m3r4c = std::array<float, 12>;
const std::string brikey = "BasicRenderInstances";

class Scene {

public:
    bool camera_changed;

    std::unordered_map<std::string, std::vector<m3r4c>> matrix_map{}; 

    inline void load_matrix_list(std::string key, std::vector<m3r4c>& matrix_list) {
        matrix_map[key] = matrix_list;
    }

    nlohmann::json sceneJson;

    inline void preload_scene(const std::string& jsonString) {
        sceneJson = nlohmann::json::parse(jsonString);
        return;
    }

    void cookGeoMatrix(std::unordered_set<uint>& volmats) {

        if (!sceneJson.contains(brikey)) return;

        auto& bri = sceneJson[brikey];

        for (auto it = bri.begin(); it != bri.end(); ++it) {

            const auto geo_name = it.value()["Geom"].template get<std::string>();
            const auto geo_type = checkGeoType(geo_name);
            if (geo_type != ShaderMark::Volume) continue;

            const auto material_str = it.value().value("Material", "Default");
            const auto material_key = std::make_tuple(material_str, geo_type);

            auto shader_index = shader_indice_table[material_key];

            const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];
    
            if ( shader_ref.vbds.size() > 0 ) {

                volmats.insert(shader_index);

                auto vdb_key = shader_ref.vbds.front();
                auto vdb_ptr = _vdb_grids_cached.at(vdb_key);

                if (vdb_ptr->dirty == false) continue;

                auto ibox = vdb_ptr->grids.front()->indexedBox();

                auto imax = glm::vec3(ibox.max().x(), ibox.max().y(), ibox.max().z()); 
                auto imin = glm::vec3(ibox.min().x(), ibox.min().y(), ibox.min().z()); 

                auto diff = imax + 1.0f - imin;
                auto center = imin + diff / 2.0f;

                glm::mat4 dirtyMatrix(1.0f);
                dirtyMatrix = glm::scale(dirtyMatrix, 1.0f/diff);
                dirtyMatrix = glm::translate(dirtyMatrix, -center);

                renderObjectMatrixMap[it.key()] = glm::transpose(dirtyMatrix);
            } // count
        }
    }

    ShaderMark checkGeoType(const std::string& geo_name, glm::mat4** matrix=nullptr) {

        if (nullptr != matrix && geoMatrixMap.count(geo_name)>0) {
            *matrix = &geoMatrixMap.at(geo_name);
        }

        if (geoTypeMap.count(geo_name)) {
            return geoTypeMap.at(geo_name);
        }

        return ShaderMark::Mesh;
    }

    void prepare_mesh_gas(OptixDeviceContext& context) {

        for (auto& [key, mesh] : _meshes_) {
            if (nullptr == mesh || mesh->vertices.empty()) { continue; }
            if (!mesh->dirty) { continue; }            
            
            mesh->dirty = false;
            mesh->buildGas(context, _mtlidlut);
        }
    }

    inline void make_scene(OptixDeviceContext& context, xinxinoptix::raii<CUdeviceptr>& bufferRoot, OptixTraversableHandle& handleRoot, 
        float3 cam=make_float3( std::numeric_limits<float>::infinity() ) ) {

        m3r4c IdentityMatrix = { 
            1,0,0,0, 
            0,1,0,0, 
            0,0,1,0 };

        auto gather = [&]() {

            if (std::isinf(cam.x)) return;
            
            auto CameraSapceMatrix = IdentityMatrix;
            CameraSapceMatrix[3] -= cam.x;
            CameraSapceMatrix[7] -= cam.y;
            CameraSapceMatrix[11] -= cam.z;

            std::vector<OptixInstance> instanced {};

            if (staticRenderGroup != 0u) {
                OptixInstance opi {};
                opi.instanceId = instanced.size();
                opi.visibilityMask = EverythingMask;
                opi.traversableHandle = staticRenderGroup;
                memcpy(opi.transform, CameraSapceMatrix.data(), sizeof(float)*12);
                instanced.push_back(opi);
            }
            if (dynamicRenderGroup != 0u) {
                OptixInstance opi {};
                opi.instanceId = instanced.size();
                opi.visibilityMask = EverythingMask;
                opi.traversableHandle = dynamicRenderGroup;
                memcpy(opi.transform, CameraSapceMatrix.data(), sizeof(float)*12);
                instanced.push_back(opi);
            }

            xinxinoptix::buildIAS(context, instanced, bufferRoot, handleRoot);
        };

        if (cam.x != std::numeric_limits<float>::infinity()) { 
            gather();
            return; 
        }

        staticRenderGroup = 0;
        dynamicRenderGroup = 0;

        nodeCache = {};
        nodeDepthCache = {};

        prepare_mesh_gas(context);
        prepare_sphere_gas(context);
        if (!sceneJson.contains(brikey)) return;

        matrix_map[""] = std::vector<m3r4c> { IdentityMatrix };

        std::unordered_map<std::string, uint64_t> candidates{};
        std::unordered_map<std::string, uint32_t> candidates_sbt{};
        std::unordered_map<std::string, VisibilityMask> candidates_mark{};

        std::unordered_map<std::string, glm::mat4*> candidates_matrix{};

        std::unordered_map<std::string, std::set<std::string>> geo_to_candidate {};
        
        const auto& bri = sceneJson[brikey];
        for (auto it = bri.begin(); it != bri.end(); ++it) {
            //std::cout << it.key() << " : " << it.value() << "\n";
            const auto candi_name = it.key();
            candidates.insert({candi_name, 0u});
            
            const auto geo_name = it.value()["Geom"].template get<std::string>();
            glm::mat4* matrix_ptr = nullptr;
            const auto geo_type = checkGeoType(geo_name, &matrix_ptr);

            candidates_matrix[candi_name] = matrix_ptr;

            const auto material_str = it.value().value("Material", "Default");
            auto material_key = std::make_tuple(material_str, geo_type);

            auto shader_index = 0u;
            auto shader_visiable = VisibilityMask::NothingMask;

            if (shader_indice_table.count(material_key)) {
                shader_index = shader_indice_table.at(material_key);
                shader_visiable = VisibilityMask::DefaultMatMask;
            }
            if (ShaderMark::Mesh == geo_type) {
                shader_index = 0u;
                shader_visiable = VisibilityMask::DefaultMatMask;
                
            } else if (ShaderMark::Volume == geo_type) {
                shader_visiable = VisibilityMask::VolumeMatMask;
                
                auto shader_index = shader_indice_table[material_key];

                const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];

                if ( shader_ref.vbds.size() > 0 ) {
            
                    auto vdb_key = shader_ref.vbds.front();

                    if (_vdb_grids_cached.count(vdb_key)==0) {
                        shader_visiable = VisibilityMask::NothingMask;
                    } else {
                        auto vdb_ptr = _vdb_grids_cached.at(vdb_key);
                        candidates[candi_name] = vdb_ptr->node->handle;
                    } //vdb_ptr
                }
            }

            candidates_sbt.insert( {candi_name, shader_index * RAY_TYPE_COUNT} );
            candidates_mark.insert( {candi_name, shader_visiable} );

            if (geo_to_candidate.count(geo_name)) {
                geo_to_candidate[geo_name].insert(candi_name);
            } else {
                geo_to_candidate[geo_name] = { candi_name };
            }
        } //bri

        auto prepare = [&candidates, &geo_to_candidate](auto& themap) {

            auto iterator = themap.begin();
            while (iterator != themap.end()) {
                auto key = iterator->first;
                if (geo_to_candidate.count(key) == 0) {
                    iterator = themap.erase(iterator);
                } else {
                    for (auto& ele : geo_to_candidate[key]) {
                        if (candidates[ele] != 0) continue;
                        candidates[ele] = iterator->second->node->handle;
                    }
                    ++iterator;
                }
            }
        };

        prepare(_meshes_);
        prepare(_spheres_);
        prepare(_sphere_groups_);
        prepare(_vol_boxs);

        std::function<OptixTraversableHandle(std::string, nlohmann::json& renderGroup, bool cache, uint& depth)> treeLook;  
        
        treeLook = [&](std::string obj_key, nlohmann::json& renderGroup, bool cache, uint& test_depth) -> OptixTraversableHandle {

            if (candidates.count(obj_key)) { //leaf node
                test_depth = 0;
                return candidates[obj_key];
            }

            if (!renderGroup.contains(obj_key)) { return 0; }
            
            if (nodeCache.count(obj_key)) {
                test_depth = nodeDepthCache[obj_key];
                return nodeCache[obj_key]->handle;
            }

            auto& ref = renderGroup[obj_key];
            std::vector<OptixInstance> instanced {}; 
            uint instanceId = 0;
            uint maxDepth = 0;

            for (auto& item : ref.items()) {
                auto item_key = item.key();

                OptixTraversableHandle handle;
                uint theDepth = 0;

                handle = treeLook(item_key, renderGroup, true, theDepth);
                maxDepth = max(maxDepth, theDepth);

                const bool leaf = candidates.count(item_key) > 0;
                const glm::mat4* matrix_ptr = candidates_matrix.count(item_key)>0? candidates_matrix.at(item_key) : nullptr;

                uint sbtOffset = 0u;

                auto vMask = EverythingMask;

                if (leaf) {
                    sbtOffset = candidates_sbt.at(item_key);
                    vMask = candidates_mark.at(item_key);
                }

                if (handle == 0u) { continue; }

                auto& matrix_keys = item.value();

                if (matrix_keys.empty()) {
                    matrix_keys = { "" };
                }

                for (auto& matrix_key : matrix_keys.items()) {

                    const auto& matrix_list = matrix_map[matrix_key.value()];
                    instanced.reserve(instanced.size() + matrix_list.size());

                    for (size_t i=0; i<matrix_list.size(); ++i) {

                        OptixInstance opi {};

                        opi.sbtOffset = sbtOffset;
                        opi.instanceId = i;
                        opi.visibilityMask = vMask;
                        opi.traversableHandle = handle;

                        if (nullptr != matrix_ptr) {

                            glm::mat4 geo_matrix(1.0f);
                            memcpy(glm::value_ptr(geo_matrix), matrix_ptr, sizeof(float)*16);

                            if (renderObjectMatrixMap.count(item_key)) {

                                auto objectMatrix = renderObjectMatrixMap.at(item_key);
                                geo_matrix = objectMatrix * geo_matrix;
                            }

                            glm::mat4 the_matrix(1.0f);
                            memcpy(glm::value_ptr(the_matrix), matrix_list[i].data(), sizeof(float)*12);

                            the_matrix = geo_matrix * the_matrix;

                            //auto dummy = glm::transpose(the_matrix);
                            auto dummy_ptr = glm::value_ptr( the_matrix );
                            memcpy(opi.transform, dummy_ptr, sizeof(float)*12);
                            
                        } else {
                            memcpy(opi.transform, matrix_list[i].data(), sizeof(float)*12);
                        }
                        instanced.push_back(opi);
                    }
                }
            }
            if (!cache) { return 0ull; }

            auto node = std::make_shared<SceneNode>();
            xinxinoptix::buildIAS(context, instanced, node->buffer, node->handle);
            nodeCache[obj_key] = node;

            test_depth = maxDepth+1;
            nodeDepthCache[obj_key] = test_depth;
            return node->handle;
        };

        auto groupTask = [&](std::string key) -> OptixTraversableHandle {

            if (!sceneJson.contains(key)) return 0;
            auto& rg = sceneJson[key];

            for (auto& item : rg.items()) {

                auto obj_key = item.key();

                uint depth = 0;
                auto handle = treeLook(obj_key, rg, false, depth);

//                std::cout << "Fetching handle: " << handle << std::endl;
            } //rg

            std::vector<OptixInstance> instanced {};
            uint instanceId = 0;

            uint the_depth = 0;
            for (auto& item : rg.items()) {

                auto obj_key = item.key();
                if (nodeDepthCache[obj_key]>0) { continue; } // deeper node
                
                uint depth = 0u;
                auto handle = treeLook(obj_key, rg, true, depth);
                the_depth = max(the_depth, depth);

                OptixInstance opi {};
                opi.instanceId = instanceId++;
                opi.visibilityMask = EverythingMask;
                opi.traversableHandle = handle;

                memcpy(opi.transform, IdentityMatrix.data(), sizeof(float)*12);
                instanced.push_back(opi);
            } //srg

            auto node = std::make_shared<SceneNode>();
            xinxinoptix::buildIAS(context, instanced, node->buffer, node->handle);
            nodeCache[key] = node;

            maxNodeDepth = max(maxNodeDepth, the_depth+1);
            return node->handle;
        };

        maxNodeDepth = 1;
        staticRenderGroup = groupTask("StaticRenderGroups");
        dynamicRenderGroup = groupTask("DynamicRenderGroups");
        maxNodeDepth = maxNodeDepth+2;
        gather();
    }

    uint maxNodeDepth = 1;
    std::unordered_map<std::string, uint> nodeDepthCache {};
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCache {};

    uint64_t staticRenderGroup {};
    uint64_t dynamicRenderGroup {};
    
    std::map<std::string, MeshDat> drawdats;

    std::map<std::string, uint> _mtlidlut;
    std::set<std::string> uniqueMatsForMesh;

    std::map<shader_key_t, uint> shader_indice_table;

    std::unordered_map<std::string, std::shared_ptr<MeshObject>> _meshes_;

    void load_object(std::string const &key, std::string const &mtlid, const std::string &instID,
                 float const *verts, size_t numverts, uint const *tris, size_t numtris,
                 std::map<std::string, std::pair<float const *, size_t>> const &vtab,
                 int const *matids, std::vector<std::string> const &matNameList) 
    {
        MeshDat &dat = drawdats[key];
        dat.dirty = true;

        dat.triMats.assign(matids, matids + numtris);
        dat.mtlidList = matNameList;
        dat.mtlid = mtlid;
        dat.instID = instID;
        dat.verts.assign(verts, verts + numverts * 3);
        dat.tris.assign(tris, tris + numtris * 3);
        //TODO: flatten just here... or in renderengineoptx.cpp
        for (auto const &[key, fptr]: vtab) {
            dat.vertattrs[key].assign(fptr.first, fptr.first + numverts * fptr.second);
        }

        uniqueMatsForMesh.insert(dat.mtlid);
        for(auto& s : dat.mtlidList) {
            uniqueMatsForMesh.insert(s);
        }
        updateGeoType(key, ShaderMark::Mesh);
    }

    void unload_object(std::string const &key) {
        drawdats.erase(key);
    }

    void updateStaticDrawObjects();
    void updateDrawObjects();

    void updateMeshMaterials(std::map<std::string, uint> const &mtlidlut) {

        _mtlidlut = mtlidlut;
        camera_changed = true;
        updateDrawObjects();
    }

std::unordered_map<std::string, ShaderMark>  geoTypeMap;
std::unordered_map<std::string, glm::mat4> geoMatrixMap;
std::unordered_map<std::string, glm::mat4> renderObjectMatrixMap;

void updateGeoType(const std::string& key, ShaderMark mark) {
    if (geoTypeMap.count(key) > 0) {
        const auto cached_type = geoTypeMap[key];
        if (cached_type == mark) return; 

        if (cached_type == ShaderMark::Mesh)
            _meshes_.erase(key);
        if (cached_type == ShaderMark::Sphere) {
            _spheres_.erase(key);
            _sphere_groups_.erase(key);
        }
        if (cached_type == ShaderMark::Volume) {
            _vol_boxs.erase(key);
        }
    }
    geoTypeMap[key] = mark;
}

std::shared_ptr<SceneNode> uniform_sphere_gas{};
std::unordered_map<std::string, std::shared_ptr<SphereTransformed>> _spheres_;

    void preload_sphere_transformed(const std::string &key, std::string const &mtlid, const std::string &instID, const glm::mat4& transform) 
    {
        auto dsphere = std::make_shared<SphereTransformed>();

        dsphere->materialID = mtlid;
        dsphere->instanceID = instID;

        auto trans = glm::transpose(transform);
        dsphere->optix_transform = trans;
        _spheres_[key] = dsphere;

        geoMatrixMap[key] = trans;
        updateGeoType(key, ShaderMark::Sphere);
    }

std::unordered_map<std::string, std::shared_ptr<SphereGroup>> _sphere_groups_;

    void preload_sphere_group(const std::string& key, std::vector<zeno::vec3f>& centerV, std::vector<float>& radiusV, std::vector<zeno::vec3f>& colorV) {
        
        auto& group = _sphere_groups_[key];
        if (nullptr == group) {
            group = std::make_shared<SphereGroup>();
            group->node = std::make_shared<SceneNode>();
        }
        group->centerV = std::move(centerV);
        group->radiusV = std::move(radiusV);
        group->colorV = std::move(colorV);

        updateGeoType(key, ShaderMark::Sphere);
    }

    void prepare_sphere_gas(OptixDeviceContext& context) {

        if (nullptr == uniform_sphere_gas) {
            uniform_sphere_gas = std::make_shared<SceneNode>();
            buildUnitSphereGAS(context, uniform_sphere_gas->handle, uniform_sphere_gas->buffer);
        }
        for (auto& [k, v] : _spheres_) {
            if (!v->dirty) continue;

            v->dirty = false;
            v->node = uniform_sphere_gas;
        }

        for (auto& [k, v] : _sphere_groups_) {
            if (!v->dirty) continue;

            v->dirty = false;
            buildSphereGroupGAS(context, *v);
        }
    }

    std::set<shader_key_t> prepareShaderSet() {

        std::set<shader_key_t> shader_key_set;

        if (sceneJson.contains(brikey)) {
            auto& bri = sceneJson[brikey];

            for (auto it = bri.begin(); it != bri.end(); ++it) {

                std::string geo_name = it.value()["Geom"].template get<std::string>();
                const auto geo_type = checkGeoType(geo_name);

                const auto material_str = it.value().value("Material", "Default");
                auto material_key = std::make_tuple(material_str, geo_type);

                shader_key_set.insert( material_key );
            }
        }

        for (auto& mat : uniqueMatsForMesh) {
            shader_key_set.insert( std::tuple {mat, ShaderMark::Mesh} );
        }

        return shader_key_set;
    }

    std::map<std::string, std::shared_ptr<VolumeWrapper>> _vdb_grids_cached;
    std::map<std::string, std::shared_ptr<VolumeWrapper>> _vol_boxs;

    bool preloadVolumeBox(const std::string& key, std::string& matid, uint8_t bounds, glm::mat4& transform) {

        auto& volume_ptr = _vol_boxs[key];
        if (nullptr == volume_ptr) {
            volume_ptr = std::make_shared<VolumeWrapper>(); 
        }
        auto trans = glm::transpose(transform);

        volume_ptr->dirty = true;
        volume_ptr->bounds = bounds;
        volume_ptr->transform = trans;
        //buildVolumeAccel(*volume_ptr, context);
        updateGeoType(key, ShaderMark::Volume);
        geoMatrixMap[key] = trans;
        return true;
    }
    
    bool processVolumeBox(OptixDeviceContext& context) {

        for (const auto& [key, vbox] : _vol_boxs) {
            if (!vbox->dirty) continue;
            vbox->dirty = false;

            buildVolumeAccel(*vbox, context);
        }
        return true;
    }

    void prepareVolumeAssets() {

        if (_vdb_grids_cached.empty()) { return; }

        std::unordered_set<uint> volmats{};
        cookGeoMatrix(volmats);

        std::unordered_set<std::string> required {};

        for(auto shader_index : volmats) {

            const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];
            if ( shader_ref.vbds.size() == 0 ) { continue; }

            for (const auto& vdb_key : shader_ref.vbds) {
                required.insert(vdb_key);
            }    
        }

        for (auto const& [key, vol] : _vdb_grids_cached) {

            if (required.count(key)) {
                if (false == vol->dirty) continue;
                // UPLOAD to GPU
                for (auto& task : vol->tasks) {
                    task();
                } //val->uploadTasks.clear();
            } else {      
                cleanupVolume(*vol); // Remove from GPU-RAM, but keep in SYS-RAM 
            }
        }
    
        for (const auto& [key, vol] : _vdb_grids_cached) {
            if (false == vol->dirty) continue;
            vol->dirty = false;

            buildVolumeAccel( *vol, OptixUtil::context );
        }
    }
    
    bool preloadVDB(const zeno::TextureObjectVDB& texVDB, std::string& combined_key);
};


inline Scene defaultScene;