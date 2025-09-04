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
#include "optix_types.h"
#include "zeno/utils/vec.h"
#include "zeno/extra/SceneAssembler.h"

#include "optixSphere.h"
#include "optixTriMesh.h"
#include "curve/optixCurve.h"

#include "LightsWrapper.h"

#include <parallel_hashmap/phmap.h>

using m3r4c = std::array<float, 12>;

const m3r4c IdentityMatrix = { 
    1,0,0,0, 
    0,1,0,0, 
    0,0,1,0 };

const std::string brikey = "BasicRenderInstances";
using Json = nlohmann::json;

class OptixScene {

private:
    phmap::parallel_flat_hash_map_m<std::string, std::function<uint64_t(OptixDeviceContext&)>> dirtyTasks;
    phmap::parallel_flat_hash_map_m<std::string, std::function<void(const std::string&)>>      cleanTasks;
    std::set<OptixUtil::TexKey> dirtyTextures;

    phmap::parallel_flat_hash_map_m<std::string, ShaderMark>  geoTypeMap;
    phmap::parallel_flat_hash_map_m<std::string, glm::mat4> geoMatrixMap;
    phmap::parallel_flat_hash_map_m<std::string, glm::mat4> renderObjectMatrixMap;

    nlohmann::json sceneJson;
    phmap::parallel_flat_hash_map_m<std::string, std::vector<m3r4c>> matrix_map{};
    phmap::parallel_flat_hash_map_m<std::string, std::vector<int>> instance_ids_map{};

    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<Hair>> hairCache;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<CurveGroupWrapper>> hairStateCache;

    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<CurveGroup>> curveGroupCache;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<CurveGroupWrapper>> curveGroupStateCache;

    std::shared_ptr<SceneNode> uniform_sphere_gas;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<SphereTransformed>> _spheres_;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<SphereGroup>> _sphere_groups_;

    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<VolumeWrapper>> _vboxs_;

    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<MeshObject>> _meshes_;

    phmap::parallel_node_hash_set_m<std::string> meshesDirty;
    phmap::parallel_node_hash_map_m<std::string, MeshDat> meshdats;
    phmap::parallel_flat_hash_set_m<std::string> uniqueMatsForMesh;

    std::unordered_map<std::string, uint64_t> gas_handles;

    uint16_t mesh_sbt_max = 0;
    std::unordered_map<shader_key_t, uint16_t, ByShaderKey> shader_indice_table;
    
public:
    phmap::parallel_flat_hash_map_m<std::string, std::pair<glm::vec3, glm::vec3>> mesh_bbox;
    phmap::parallel_flat_hash_map_m<std::string, std::vector<glm::mat4>> glm_matrix_map;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<VolumeWrapper>> _vdb_grids_cached;

    inline void load_shader_indice_table(std::unordered_map<shader_key_t, uint16_t, ByShaderKey> &table) {
        shader_indice_table = table;

        for (const auto& [k, v] : shader_indice_table) {
            const auto& [_, mark] = k;
            if (mark == ShaderMark::Mesh) mesh_sbt_max = max(mesh_sbt_max, v);
        }
        for (const auto &[key, value]: shader_indice_table) {
            dc_index_to_mat[value] = std::get<0>(key);
        }
    }
    inline void load_matrix_list_to_glm(const std::string &key, zeno::PrimitiveObject *prim) {
        size_t count = prim->verts.size() / 4;
        if (count == 0) {
            return;
        }
        std::vector<glm::mat4> matrixs(count);
        for (auto i = 0; i < count; i++) {
            auto &matrix = matrixs[i];
            matrix[3][3] = 1;
            auto &r0 = matrix[0];
            auto &r1 = matrix[1];
            auto &r2 = matrix[2];
            auto &t = matrix[3];
            r0[0] = prim->verts[0 + i * 4][0];
            r1[0] = prim->verts[0 + i * 4][1];
            r2[0] = prim->verts[0 + i * 4][2];
            t[0]  = prim->verts[1 + i * 4][0];
            r0[1] = prim->verts[1 + i * 4][1];
            r1[1] = prim->verts[1 + i * 4][2];
            r2[1] = prim->verts[2 + i * 4][0];
            t[1]  = prim->verts[2 + i * 4][1];
            r0[2] = prim->verts[2 + i * 4][2];
            r1[2] = prim->verts[3 + i * 4][0];
            r2[2] = prim->verts[3 + i * 4][1];
            t[2]  = prim->verts[3 + i * 4][2];
        }
        glm_matrix_map[key] = matrixs;
    }

    inline void load_matrix_list(std::string key, std::vector<m3r4c>& matrix_list, std::vector<int> instance_ids) {
        matrix_map[key] = std::move(matrix_list);
        if (instance_ids.size() > 0) {
            instance_ids_map[key] = std::move(instance_ids);
        }
    }    

    std::unordered_map<uint64_t, std::string> gas_to_obj_id;
    std::unordered_map<uint64_t, std::string> dc_index_to_mat;

    Json static_scene_tree;
    Json dynamic_scene_tree;
    std::shared_ptr<zeno::SceneObject> dynamic_scene = std::make_shared<zeno::SceneObject>();
    std::unordered_map<std::string, glm::mat4> modified_xfroms;
    std::optional<std::tuple<std::string, glm::mat4, glm::mat4>> cur_node;
    std::vector<std::string> cur_link;

    inline void preload_scene(const std::string& jsonString) {
        try {
            sceneJson = nlohmann::json::parse(jsonString);
        }
        catch (...) {
            zeno::log_error("Can not parse json in preload_scene");
        }
    }

    inline void updateGeoType(const std::string& key, ShaderMark mark) {
        geoTypeMap.insert( {key, mark} );
    }

    void cookGeoMatrix(std::unordered_set<uint>& volmats) {

        if (!sceneJson.contains(brikey)) return;

        auto& bri = sceneJson[brikey];

        for (auto it = bri.begin(); it != bri.end(); ++it) {

            const auto geo_name = it.value()["Geom"].template get<std::string>();
            const auto geo_type = checkGeoType(geo_name);
            if (geo_type != ShaderMark::Volume) continue;

            const auto material_str = it.value().value("Material", "");
            const auto material_key = std::make_tuple(material_str, geo_type);

            auto shader_index = shader_indice_table[material_key];

            const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];
    
            if ( shader_ref.vbds.size() > 0 ) {

                volmats.insert(shader_index);

                auto vdb_key = shader_ref.vbds.front();
                if (_vdb_grids_cached.count(vdb_key) == 0) continue;

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

    LightsWrapper lightsWrapper;

    void prepare_light_ias(OptixDeviceContext& context) {

        std::vector<OptixInstance> optix_instances;

        if (lightsWrapper.lightTrianglesGas != 0)
        {
            OptixInstance opinstance {};

            auto combinedID = std::tuple(std::string("Light"), ShaderMark::Mesh);
            auto shader_index = shader_indice_table[combinedID];

            opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
            opinstance.instanceId = 0;
            opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
            opinstance.visibilityMask = LightMatMask;
            opinstance.traversableHandle = lightsWrapper.lightTrianglesGas;
            memcpy(opinstance.transform, IdentityMatrix.data(), sizeof(float) * 12);

            optix_instances.push_back( opinstance );
        }

        if (lightsWrapper.lightPlanesGas != 0)
        {
            OptixInstance opinstance {};

            auto combinedID = std::tuple(std::string("Light"), ShaderMark::Mesh);
            auto shader_index = shader_indice_table[combinedID];

            opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
            opinstance.instanceId = 1;
            opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
            opinstance.visibilityMask = LightMatMask;
            opinstance.traversableHandle = lightsWrapper.lightPlanesGas;
            memcpy(opinstance.transform, IdentityMatrix.data(), sizeof(float) * 12);

            optix_instances.push_back( opinstance );
        }

        if (lightsWrapper.lightSpheresGas != 0)
        {
            OptixInstance opinstance {};

            auto combinedID = std::tuple(std::string("Light"), ShaderMark::Sphere);
            auto shader_index = shader_indice_table[combinedID];

            opinstance.flags = OPTIX_INSTANCE_FLAG_NONE;
            opinstance.instanceId = 2;
            opinstance.sbtOffset = shader_index * RAY_TYPE_COUNT;
            opinstance.visibilityMask = LightMatMask;
            opinstance.traversableHandle = lightsWrapper.lightSpheresGas;
            memcpy(opinstance.transform, IdentityMatrix.data(), sizeof(float) * 12);

            optix_instances.push_back( opinstance );
        }

        xinxinoptix::buildIAS(context, optix_instances, lightsWrapper.lightIasBuffer, lightsWrapper.lightIasHandle);
    }

    struct Candidate {
        uint64_t handle;
        uint32_t sbt;
        glm::mat4* matrix{};
        VisibilityMask vmask;
    };

    inline void make_scene(OptixDeviceContext& context, xinxinoptix::raii<CUdeviceptr>& bufferRoot, OptixTraversableHandle& handleRoot, 
        float3 cam=make_float3( std::numeric_limits<float>::infinity() ) ) {

        auto gather = [&]() {

            if (std::isinf(cam.x)) { cam = {}; }
            
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

            if (lightsWrapper.lightIasHandle != 0u) {
                OptixInstance opi {};
                opi.instanceId = instanced.size();
                opi.visibilityMask = LightMatMask;
                opi.traversableHandle = lightsWrapper.lightIasHandle;
                memcpy(opi.transform, CameraSapceMatrix.data(), sizeof(float)*12);
                instanced.push_back(opi);

                maxNodeDepth = max(maxNodeDepth, 3u);
            }

            xinxinoptix::buildIAS(context, instanced, bufferRoot, handleRoot);
        };

        if (cam.x != std::numeric_limits<float>::infinity() || !sceneJson.contains(brikey) ) { 
            gather();
            return;
        }

        dynamicRenderGroup = 0;
        nodeCache = {};
        nodeDepthCache = {};

        //if (false) {
//            staticRenderGroup = 0;
//            nodeCacheStatic = {};
//            nodeDepthCacheStatic = {};
        //}

        if (!dirtyTasks.empty()) {
            
            for(auto& [key, task] : dirtyTasks) {
                auto handle = task(context);
                gas_handles[key] = handle;
                gas_to_obj_id[handle] = key;
            }
            dirtyTasks.clear();
        }
        if (!dirtyTextures.empty()) {
            for (auto& key : dirtyTextures) {
                auto& ref = OptixUtil::tex_lut.at(key);
                if (ref.use_count()==1) {
                    OptixUtil::removeTexture(key);
                }
            }
            dirtyTextures.clear();
        }

        matrix_map[""] = std::vector<m3r4c> { IdentityMatrix };
        static const std::vector fallback_keys { "" };

        std::unordered_map<std::string, Candidate> candidates {};
        
        const auto& bri = sceneJson[brikey];
        for (auto it = bri.begin(); it != bri.end(); ++it) {
            //std::cout << it.key() << " : " << it.value() << "\n";
            const auto candi_name = it.key();
            const auto geo_name = it.value()["Geom"].template get<std::string>();

            cleanTasks.erase(geo_name);

            glm::mat4* matrix_ptr = nullptr;
            const auto geo_type = checkGeoType(geo_name, &matrix_ptr);

            Candidate candi {};
            candi.handle = gas_handles[geo_name];
            candi.matrix = matrix_ptr;

            const auto material_str = it.value().value("Material", "");
            auto material_key = std::make_tuple(material_str, geo_type);

            uint16_t shader_index = 0u;
            auto shader_visiable = VisibilityMask::NothingMask;

            if (shader_indice_table.count(material_key)) {
                shader_index = shader_indice_table.at(material_key);
                shader_visiable = VisibilityMask::DefaultMatMask;
            }
            if (ShaderMark::Mesh == geo_type) {

                auto mesh = _meshes_[geo_name];
                if (mesh != nullptr && mesh->mat_idx.size()==1) {
                    shader_index = mesh->mat_idx[0];
                } else {
                    shader_index = 0u;
                }
                shader_visiable = VisibilityMask::DefaultMatMask;
                
            } else if (ShaderMark::Volume == geo_type) {
                shader_visiable = VisibilityMask::VolumeMaskHeterogeneous;
                
                auto shader_index = shader_indice_table[material_key];

                const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];

                if ( shader_ref.vbds.size() > 0 ) {
            
                    auto vdb_key = shader_ref.vbds.front();

                    if (_vdb_grids_cached.count(vdb_key)==0) {
                        shader_visiable = VisibilityMask::NothingMask;
                    } else {
                        auto vdb_ptr = _vdb_grids_cached.at(vdb_key);
                        candi.handle = vdb_ptr->node->handle;
                    } //vdb_ptr
                } 
                if (shader_ref.isHomoVol())
                    shader_visiable = VisibilityMask::VolumeMaskAnalytics;
            }

            candi.sbt = shader_index * RAY_TYPE_COUNT;
            candi.vmask = shader_visiable;
            candidates[candi_name] = candi;
        } //bri

        if (!cleanTasks.empty()) {
            for (auto& [k, task] : cleanTasks) {
                task(k);
                gas_handles.erase(k);
            }
            cleanTasks.clear();
        }

        std::function<OptixTraversableHandle(std::string&, nlohmann::json& renderGroup, bool cache, uint& test_depth, 
                                                decltype(nodeCache)& nodeCache, decltype(nodeDepthCache)& nodeDepthCache)> treeLook;  

        treeLook = [this, &treeLook, &context, &candidates]
                        (std::string& obj_key, nlohmann::json& renderGroup, bool cache, uint& test_depth, 
                            decltype(nodeCache)& nodeCache, decltype(nodeDepthCache)& nodeDepthCache) -> OptixTraversableHandle 
        {
            if (candidates.count(obj_key)) { //leaf node
                test_depth = 1;
                return candidates[obj_key].handle;
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

                handle = treeLook(item_key, renderGroup, true, theDepth, nodeCache, nodeDepthCache);
                maxDepth = max(maxDepth, theDepth);

                const bool leaf = candidates.count(item_key) > 0;
               
                uint sbtOffset = 0u;
                auto vMask = EverythingMask;
                glm::mat4* matrix_ptr = nullptr;

                if (leaf) {
                    auto& candi = candidates.at(item_key);
                    sbtOffset = candi.sbt;
                    vMask = candi.vmask;
                    matrix_ptr = candidates.at(item_key).matrix;
                }

                if (handle == 0u) { continue; }

                auto& matrix_keys = item.value();
                if (matrix_keys.empty()) {
                    matrix_keys = fallback_keys;
                }

                for (auto& matrix_key : matrix_keys.items()) {

                    const auto& matrix_list = matrix_map[matrix_key.value()];
                    const auto id_it = instance_ids_map.find(matrix_key.value());
                    const auto has_custom_id = id_it != instance_ids_map.end();

                    instanced.reserve(instanced.size() + matrix_list.size());

                    for (size_t i=0; i<matrix_list.size(); ++i) {

                        OptixInstance opi {};

                        opi.sbtOffset = sbtOffset;
                        if (has_custom_id) {
                            opi.instanceId = id_it->second[i];
                        }
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

        auto groupTask = [&](std::string key, decltype(nodeCache)& nodeCache, decltype(nodeDepthCache)& nodeDepthCache) -> OptixTraversableHandle 
        {
            if (!sceneJson.contains(key)) return 0;
            auto& rg = sceneJson[key];

            for (auto& item : rg.items()) {

                auto obj_key = item.key();

                uint depth = 0;
                auto handle = treeLook(obj_key, rg, false, depth, nodeCache, nodeDepthCache);
            } //rg

            std::vector<OptixInstance> instanced {};
            uint instanceId = 0;

            uint the_depth = 0;
            for (auto& item : rg.items()) {

                auto obj_key = item.key();
                if (nodeDepthCache[obj_key]>0) { continue; } // deeper node
                
                uint depth = 0u;
                auto handle = treeLook(obj_key, rg, true, depth, nodeCache, nodeDepthCache);
                the_depth = max(the_depth, depth);

                OptixInstance opi {};
                opi.instanceId = instanceId++;
                opi.visibilityMask = EverythingMask;
                opi.traversableHandle = handle;

                memcpy(opi.transform, IdentityMatrix.data(), sizeof(float)*12);
                instanced.push_back(opi);
            } //srg

            if (instanced.size() == 0) return 0;

            auto node = std::make_shared<SceneNode>();
            xinxinoptix::buildIAS(context, instanced, node->buffer, node->handle);
            nodeCache[key] = node;
            nodeDepthCache[key] = the_depth+1;
            assert(node->handle!=0);
            
            return node->handle;
        };

        maxNodeDepth = 1;
        if (0 == staticRenderGroup) {
            staticRenderGroup = groupTask("StaticRenderGroups", nodeCacheStatic, nodeDepthCacheStatic);
        }
        dynamicRenderGroup = groupTask("DynamicRenderGroups", nodeCache, nodeDepthCache);
        maxNodeDepth = max(nodeDepthCacheStatic["StaticRenderGroups"], nodeDepthCache["DynamicRenderGroups"]);
        maxNodeDepth += 1;
        gather();
    }

    uint maxNodeDepth = 1;
    std::unordered_map<std::string, uint> nodeDepthCache {};
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCache {};
    std::unordered_map<std::string, uint> nodeDepthCacheStatic {};
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCacheStatic {};

    uint64_t staticRenderGroup {};
    uint64_t dynamicRenderGroup {};

    void preload_mesh(std::string const &key, std::string const &mtlid,
                 float const *verts, size_t numverts, uint const *tris, size_t numtris,
                 std::map<std::string, std::pair<float const *, size_t>> const &vtab,
                 int const *matids, std::vector<std::string> const &matNameList);

    void unload_object(std::string const &key) {
        if (cleanTasks.count(key)==0) return;
        cleanTasks[key](key);
        cleanTasks.erase(key);
    }

    void updateDrawObjects(uint16_t sbt_count);

    void updateMeshMaterials() {
        updateDrawObjects(mesh_sbt_max+1);
    }

    void preloadHair(const std::string& name, const std::string& filePath, uint mode, glm::mat4 transform=glm::mat4(1.0f));
    void preloadCurveGroup(std::vector<float3>& points, std::vector<float>& widths, std::vector<float3>& normals, std::vector<uint>& strands, zeno::CurveType curveType, const std::string& key); 
    
    void preload_sphere(const std::string &key, const glm::mat4& transform);
    void preload_sphere_group(const std::string& key, std::vector<zeno::vec3f>& centerV, std::vector<float>& radiusV, std::vector<zeno::vec3f>& colorV);

    auto prepareShaderSet() {

        std::map<std::string, std::set<ShaderMark>> shader_key_set;

        uniqueMatsForMesh.insert("");
        for (auto& mat : uniqueMatsForMesh) {
            auto& cached = shader_key_set[mat];
            cached.insert( ShaderMark::Mesh );
        }

        if (sceneJson.contains(brikey)) {
            auto& bri = sceneJson[brikey];

            for (auto it = bri.begin(); it != bri.end(); ++it) {

                std::string geo_name = it.value()["Geom"].template get<std::string>();
                const auto geo_type = checkGeoType(geo_name);

                const auto material_str = it.value().value("Material", "");
                auto material_key = std::make_tuple(material_str, geo_type);

                if (shader_key_set.count(material_str)==0) {
                    shader_key_set[material_str] = std::set<ShaderMark>();
                }
                shader_key_set[material_str].insert(geo_type);
            }
        }

        return shader_key_set;
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

    bool preloadVolumeBox(const std::string& key, std::string& matid, uint8_t bounds, glm::mat4& transform);
    bool preloadVDB(const zeno::TextureObjectVDB& texVDB, std::string& combined_key);
};


inline OptixScene defaultScene;