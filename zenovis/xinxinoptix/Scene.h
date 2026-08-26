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
#include <sstream>

#include <tuple>
#include <array>
#include <algorithm>
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

#include "optixCommon.h"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/fwd.hpp"
#include "magic_enum.hpp"
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
    std::optional<std::string> sceneJsonSource;
    bool candidateBindingsDirty {true};
    bool sceneGraphDirty {true};
    std::unordered_map<std::string, std::vector<std::string>> dynamicMatrixDependencies;

    phmap::parallel_node_hash_set_m<std::string> matrix_dirty;
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
    phmap::parallel_flat_hash_set_m<std::string> uniqueMatsForMesh;

    std::unordered_map<std::string, uint64_t> gas_handles;

    uint16_t mesh_sbt_max = 0;
    std::unordered_map<shader_key_t, uint16_t, ByShaderKey> shader_indice_table;
    
public:
    uint32_t frameid;
    phmap::parallel_flat_hash_map_m<std::string, std::pair<glm::vec3, glm::vec3>> mesh_bbox;
    phmap::parallel_flat_hash_map_m<std::string, std::vector<glm::mat4>> glm_matrix_map;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<VolumeWrapper>> _vdb_grids_cached;

    inline void load_shader_indice_table(std::unordered_map<shader_key_t, uint16_t, ByShaderKey> &table) {
        const bool unchanged = table.size() == shader_indice_table.size()
            && std::all_of(table.begin(), table.end(), [&](const auto& item) {
                const auto current = shader_indice_table.find(item.first);
                return current != shader_indice_table.end() && current->second == item.second;
            });
        if (unchanged) return;

        shader_indice_table = table;
        candidateBindingsDirty = true;

        mesh_sbt_max = 0;
        dc_index_to_mat.clear();
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
        matrix_dirty.insert(key);
        if (!instance_ids.empty()) {
            instance_ids_map[key] = std::move(instance_ids);
        } else {
            // Frame data may stop supplying custom IDs. Do not retain a
            // shorter vector from an earlier frame and index through it while
            // rebuilding the IAS.
            instance_ids_map.erase(key);
        }
    }    

    std::unordered_map<uint64_t, std::string> gas_to_obj_id;
    std::unordered_map<uint64_t, std::string> dc_index_to_mat;

    Json static_scene_tree;
    Json dynamic_scene_tree;
    std::vector<std::string> lights_name;
    std::shared_ptr<zeno::SceneObject> dynamic_scene = std::make_shared<zeno::SceneObject>();
    std::unordered_map<std::string, glm::mat4> modified_xfroms;
    std::optional<std::tuple<std::string, glm::mat4, glm::mat4>> cur_node;
    std::vector<std::string> cur_link;

    inline void preload_scene(const std::string& jsonString) {
        if (sceneJsonSource && *sceneJsonSource == jsonString) return;

        try {
            auto nextScene = nlohmann::json::parse(jsonString);
            sceneJsonSource = jsonString;
            if (nextScene == sceneJson) return;

            sceneJson = std::move(nextScene);
            rebuildDynamicMatrixDependencies();
            candidateBindingsDirty = true;
            sceneGraphDirty = true;
        }
        catch (...) {
            zeno::log_error("Can not parse json in preload_scene");
        }
    }

    inline void updateGeoType(const std::string& key, ShaderMark mark) {
        const auto [it, inserted] = geoTypeMap.insert({key, mark});
        if (inserted) {
            candidateBindingsDirty = true;
        } else if (it->second != mark) {
            it->second = mark;
            candidateBindingsDirty = true;
        }
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

            auto shader_it = shader_indice_table.find(material_key);
            if (shader_it == shader_indice_table.end()) { continue; }

            auto shader_index = shader_it->second;
            if (shader_index >= OptixUtil::rtMaterialShaders.size()) { continue; }

            const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];
    
            if ( shader_ref.vbds.size() > 0 ) {

                volmats.insert(shader_index);

                const auto density_slot = shader_ref.density_vdb_primary_slot;
                if (density_slot >= shader_ref.vbds.size()) { continue; }

                auto vdb_key = shader_ref.vbds[density_slot];
                if (_vdb_grids_cached.count(vdb_key) == 0) continue;

                auto vdb_ptr = _vdb_grids_cached.at(vdb_key);
                if (vdb_ptr->dirty == false) continue;
                
                continue;

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
        optix_instances.reserve(3);

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

        xinxinoptix::buildIAS(context, optix_instances, lightsWrapper.lightIasBuffer, lightsWrapper.lightIasHandle, false, &iasBuildWorkspace, true);
    }

    struct Candidate {
        uint64_t handle;
        uint32_t sbt;
        glm::mat4* matrix{};
        VisibilityMask vmask;

        bool operator!=(const Candidate& other) const {
            return handle!=other.handle || sbt!=other.sbt || matrix!=other.matrix || vmask!=other.vmask;
        }

        bool operator==(const Candidate& other) const {
            return !(*this != other);
        }
    };

private:
    struct TreeLookPolicy {
        bool collapseSingleChild{};
        bool allowUpdate{};
        bool preferFastBuild{};
    };

    static m3r4c composeAffine(const m3r4c& parent, const m3r4c& local) {
        m3r4c result{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                result[row * 4 + column] =
                    parent[row * 4 + 0] * local[0 * 4 + column]
                  + parent[row * 4 + 1] * local[1 * 4 + column]
                  + parent[row * 4 + 2] * local[2 * 4 + column];
            }
            result[row * 4 + 3] = parent[row * 4 + 3]
              + parent[row * 4 + 0] * local[0 * 4 + 3]
              + parent[row * 4 + 1] * local[1 * 4 + 3]
              + parent[row * 4 + 2] * local[2 * 4 + 3];
        }
        return result;
    }

    static void appendInstance(InstanceCollector& collector, OptixTraversableHandle handle, const m3r4c& transform,
                               uint32_t instance_id, uint32_t sbt_offset = 0, uint8_t visibility_mask = EverythingMask) {
        OptixInstance instance{};
        memcpy(instance.transform, transform.data(), sizeof(float) * 12);
        instance.instanceId = instance_id;
        instance.sbtOffset = sbt_offset;
        instance.visibilityMask = visibility_mask;
        instance.traversableHandle = handle;
        collector.append(instance);
    }

    template <class Visitor>
    void forEachGroupChildInstance(const nlohmann::json& group, InstanceCollector& destination, Visitor&& visit) {
        for (const auto& item : group.items()) {
            const auto& item_key = item.key();
            const auto& matrix_keys = item.value();
            const auto candidate_it = candidates.find(item_key);

            glm::mat4 geometry_transform(1.0f);
            const bool has_geometry_transform =
                candidate_it != candidates.end() && candidate_it->second.matrix != nullptr;
            if (has_geometry_transform) {
                memcpy(glm::value_ptr(geometry_transform), candidate_it->second.matrix, sizeof(float) * 16);
                const auto object_matrix_it = renderObjectMatrixMap.find(item_key);
                if (object_matrix_it != renderObjectMatrixMap.end()) {
                    geometry_transform = object_matrix_it->second * geometry_transform;
                }
            }

            auto visitMatrix = [&](const m3r4c& matrix, size_t matrix_index, const std::vector<int>* instance_ids) {
                const bool has_custom_id = instance_ids != nullptr && matrix_index < instance_ids->size();
                const uint32_t instance_id = has_custom_id? static_cast<uint32_t>((*instance_ids)[matrix_index]) : 0u;

                m3r4c local_transform = matrix;
                if (has_geometry_transform) {
                    glm::mat4 matrix_with_geometry(1.0f);
                    memcpy(glm::value_ptr(matrix_with_geometry), matrix.data(), sizeof(float) * 12);
                    matrix_with_geometry = geometry_transform * matrix_with_geometry;
                    memcpy(local_transform.data(), glm::value_ptr(matrix_with_geometry), sizeof(float) * 12);
                }
                visit(item_key, local_transform, instance_id);
            };

            if (matrix_keys.empty()) {
                destination.reserveAdditional(1);
                visitMatrix(IdentityMatrix, 0, nullptr);
                continue;
            }

            for (const auto& matrix_key_json : matrix_keys.items()) {
                const auto& matrix_key = matrix_key_json.value().get_ref<const std::string&>();
                const auto matrix_it = matrix_map.find(matrix_key);
                if (matrix_it == matrix_map.end()) continue;
                const auto& matrix_list = matrix_it->second;
                const auto id_it = instance_ids_map.find(matrix_key);
                const auto* instance_ids = id_it == instance_ids_map.end()? nullptr : std::addressof(id_it->second);
                destination.reserveAdditional(matrix_list.size());
                for (size_t i = 0; i < matrix_list.size(); ++i) {
                    visitMatrix(matrix_list[i], i, instance_ids);
                }
            }
        }
    }

    void rebuildDynamicMatrixDependencies() {
        dynamicMatrixDependencies.clear();

        const auto groups_it = sceneJson.find("DynamicRenderGroups");
        if (groups_it == sceneJson.end() || !groups_it->is_object()) return;
        const auto& groups = *groups_it;

        std::unordered_map<std::string, std::vector<std::string>> parents;
        std::unordered_map<std::string, std::unordered_set<std::string>> direct_dependencies;

        for (const auto& group : groups.items()) {
            if (!group.value().is_object()) continue;
            for (const auto& child : group.value().items()) {
                if (groups.contains(child.key())) {
                    parents[child.key()].push_back(group.key());
                }
                for (const auto& matrix : child.value().items()) {
                    if (!matrix.value().is_string()) continue;
                    const auto& matrix_key = matrix.value().get_ref<const std::string&>();
                    direct_dependencies[matrix_key].insert(group.key());
                }
            }
        }

        for (auto& [matrix_key, direct] : direct_dependencies) {
            std::unordered_set<std::string> dependencies = std::move(direct);
            std::vector<std::string> pending(dependencies.begin(), dependencies.end());
            while (!pending.empty()) {
                auto group_name = std::move(pending.back());
                pending.pop_back();

                const auto parent_it = parents.find(group_name);
                if (parent_it == parents.end()) continue;
                for (const auto& parent : parent_it->second) {
                    if (dependencies.insert(parent).second) {
                        pending.push_back(parent);
                    }
                }
            }

            auto& cached = dynamicMatrixDependencies[matrix_key];
            cached.reserve(dependencies.size());
            for (const auto& group_name : dependencies) {
                cached.push_back(group_name);
            }
        }
    }

    void invalidateDynamicMatrixDependencies() {
        for (const auto& matrix_key : matrix_dirty) {
            const auto dependency_it = dynamicMatrixDependencies.find(matrix_key);
            if (dependency_it == dynamicMatrixDependencies.end()) continue;

            for (const auto& group_name : dependency_it->second) {
                const auto node_it = nodeCache.find(group_name);
                if (node_it != nodeCache.end() && node_it->second) {
                    node_it->second->frame = UINT32_MAX;
                }
            }
        }
    }

    void refreshCandidateBindings(const std::unordered_set<std::string>& geometry_dirty, std::unordered_set<std::string>& candidate_dirty) {
        const auto& bri = sceneJson[brikey];

        for (auto candidate_it = candidates.begin(); candidate_it != candidates.end();) {
            if (!bri.contains(candidate_it->first)) {
                candidate_dirty.insert(candidate_it->first);
                candidate_it = candidates.erase(candidate_it);
            } else {
                ++candidate_it;
            }
        }

        candidates.reserve(bri.size());
        for (auto it = bri.begin(); it != bri.end(); ++it) {
            const auto& candidate_name = it.key();
            const auto& geometry_name = it.value()["Geom"].get_ref<const std::string&>();

            cleanTasks.erase(geometry_name);

            glm::mat4* matrix = nullptr;
            const auto geometry_type = checkGeoType(geometry_name, &matrix);

            Candidate candidate {};
            candidate.handle = gas_handles[geometry_name];
            candidate.matrix = matrix;

            const auto material = it.value().value("Material", "");
            const auto material_key = std::make_tuple(material, geometry_type);
            const auto shader_it = shader_indice_table.find(material_key);

            uint16_t shader_index = 0u;
            auto visibility = VisibilityMask::NothingMask;
            if (shader_it != shader_indice_table.end()) {
                shader_index = shader_it->second;
                visibility = VisibilityMask::DefaultMatMask;
            }

            if (geometry_type == ShaderMark::Mesh) {
                const auto mesh = _meshes_[geometry_name];
                shader_index = mesh != nullptr && mesh->mat_idx.size() == 1 ? mesh->mat_idx[0] : 0u;
                visibility = VisibilityMask::DefaultMatMask;
            } else if (geometry_type == ShaderMark::Volume) {
                visibility = VisibilityMask::VolumeMaskHeterogeneous;
                if (shader_it == shader_indice_table.end()
                    || shader_it->second >= OptixUtil::rtMaterialShaders.size()) {
                    visibility = VisibilityMask::NothingMask;
                } else {
                    shader_index = shader_it->second;
                    const auto& shader = OptixUtil::rtMaterialShaders[shader_index];
                    if (!shader.vbds.empty()) {
                        const auto density_slot = shader.density_vdb_primary_slot;
                        if (density_slot >= shader.vbds.size()) {
                            visibility = VisibilityMask::NothingMask;
                        } else {
                            const auto& vdb_key = shader.vbds[density_slot];
                            const auto vdb_it = _vdb_grids_cached.find(vdb_key);
                            if (vdb_it == _vdb_grids_cached.end() || vdb_it->second->node->handle == 0) {
                                visibility = VisibilityMask::NothingMask;
                            } else {
                                candidate.handle = vdb_it->second->node->handle;
                            }
                        }
                    }
                    if (shader.isHomoVol()) {
                        visibility = VisibilityMask::VolumeMaskAnalytics;
                    }
                }
            }

            candidate.sbt = shader_index * RAY_TYPE_COUNT;
            candidate.vmask = visibility;

            const auto previous = candidates.find(candidate_name);
            if (geometry_dirty.count(geometry_name) != 0
                || previous == candidates.end() || previous->second != candidate) {
                candidate_dirty.insert(candidate_name);
            }
            if (previous == candidates.end()) {
                candidates.emplace(candidate_name, candidate);
            } else {
                previous->second = candidate;
            }
        }

        candidateBindingsDirty = false;
    }

public:

    inline void make_scene(OptixDeviceContext& context, float3 cam, bool camera_only) {

        this->cameraPos = cam;
        auto gather = [&]() {
            
            auto CameraSapceMatrix = IdentityMatrix;
            CameraSapceMatrix[3] -= cam.x;
            CameraSapceMatrix[7] -= cam.y;
            CameraSapceMatrix[11] -= cam.z;

            std::vector<OptixInstance> instanced {};
            instanced.reserve(3);

            OptixInstance opi {};
            memcpy(opi.transform, CameraSapceMatrix.data(), sizeof(float)*12);

            if (lightsWrapper.lightIasHandle != 0u) {
                opi.instanceId = 0;
                opi.visibilityMask = LightMatMask;
                opi.traversableHandle = lightsWrapper.lightIasHandle;
                instanced.push_back(opi);

                maxNodeDepth = max(maxNodeDepth, 3);
            }

            if (staticRenderGroup != 0u) {
                opi.instanceId = 1;
                opi.visibilityMask = EverythingMask;
                opi.traversableHandle = staticRenderGroup;
                instanced.push_back(opi);
            }
            if (dynamicRenderGroup != 0u) {
                opi.instanceId = 2;
                opi.visibilityMask = EverythingMask;
                opi.traversableHandle = dynamicRenderGroup;
                memcpy(opi.transform, IdentityMatrix.data(), sizeof(float)*12);
                instanced.push_back(opi);
            }

            const bool same_children = rootLightChild == lightsWrapper.lightIasHandle
                                && rootStaticChild == staticRenderGroup
                                && rootDynamicChild == dynamicRenderGroup;
            xinxinoptix::buildIAS(context, instanced, this->rootNode, same_children, true);
            rootLightChild = lightsWrapper.lightIasHandle;
            rootStaticChild = staticRenderGroup;
            rootDynamicChild = dynamicRenderGroup;
        };

        if (camera_only || !sceneJson.contains(brikey)) {
            if (!nodeCache.empty())
                dynamicRenderGroup = groupTask("DynamicRenderGroups", "DynamicEntries", true, nodeCache, true);
            gather();
            return;
        }

        dynamicRenderGroup = 0;
        //nodeCache = {};

        // A child GAS UPDATE keeps its traversable handle, but OptiX still
        // requires every ancestor IAS to be updated or rebuilt. Preserve the
        // dirty task keys through the existing candidate pass so dependency
        // invalidation does not depend on the opaque handle changing.
        std::unordered_set<std::string> geometry_dirty;
        geometry_dirty.reserve(dirtyTasks.size());
        if (!dirtyTasks.empty()) {
            
            for(auto& [key, task] : dirtyTasks) {
                geometry_dirty.insert(key);
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

        std::unordered_set<std::string> candidate_dirty;
        if (candidateBindingsDirty || !geometry_dirty.empty() || !cleanTasks.empty()) {
            refreshCandidateBindings(geometry_dirty, candidate_dirty);
        }

        if (!cleanTasks.empty()) {
            for (auto& [k, task] : cleanTasks) {
                task(k);
                gas_handles.erase(k);
            }
            cleanTasks.clear();
        }

        if (sceneGraphDirty) {
            for (auto& [_, node] : nodeCache) {
                if (node) node->frame = UINT32_MAX;
            }
            for (auto& [_, node] : nodeCacheStatic) {
                if (node) node->frame = UINT32_MAX;
            }
            staticRenderGroup = 0;
            sceneGraphDirty = false;
        }

        std::function<bool(const std::string&, nlohmann::json&, decltype(nodeCache)&)> dirtyCheck;

        dirtyCheck = [&](const std::string& key, nlohmann::json& renderGroup, decltype(nodeCache)& nodeCache) -> bool {

            auto& node = nodeCache[key];
            if (nullptr == node) {
                //return false;
                node = std::make_shared<SceneNode>();
            }

            auto find_obj = renderGroup.find(key);
            if (find_obj == renderGroup.end()) {
                return candidate_dirty.count(key)>0 || candidates.count(key)==0;
            }
            auto& ref = *find_obj;

            bool dirty = false;
            for (auto& item : ref.items()) {
                auto& item_key = item.key();
                bool check = dirtyCheck(item_key, renderGroup, nodeCache);
                dirty |= check;

                auto& matrix_keys = item.value();
                for (auto& matrix_key : matrix_keys.items()) {
                    const auto& matrix_name = matrix_key.value().get_ref<const std::string&>();
                    if (matrix_dirty.contains(matrix_name)) {
                        dirty |= true; break;
                    }
                }
            }
            if (dirty) {
                node->frame = UINT32_MAX;
            }
            return dirty;
        };

        auto dirtyGroup = [&](std::string_view group_key, std::string_view entry_key, decltype(nodeCache)& nodeCache) 
        {
            if (!sceneJson.contains(group_key)) return false;
            if (!sceneJson.contains(entry_key)) return false;

            auto& rg = sceneJson[group_key];
            auto& entrys = sceneJson[entry_key];

            bool dirty = false;
            for (const auto& kv : entrys.items()) {
                dirty |= dirtyCheck(kv.key(), rg, nodeCache);
            }
            return dirty;
        };

        if (!matrix_dirty.empty() || !candidate_dirty.empty()) {
            if (candidate_dirty.empty()) {
                invalidateDynamicMatrixDependencies();
            } else {
                // Candidate/GAS changes are rare and can affect topology, so
                // retain the recursive dependency check for that general case.
                dirtyGroup("DynamicRenderGroups", "DynamicEntries", nodeCache);
            }
            if (collapseDynamicSingleChildIAS && !candidate_dirty.empty()) {
                // Candidate changes can alter the collapsed topology itself.
                for (auto& [_, node] : nodeCache) {
                    if (node) node->frame = UINT32_MAX;
                }
            }
            if (!candidate_dirty.empty() && dirtyGroup("StaticRenderGroups", "StaticEntries", nodeCacheStatic)) {
                staticRenderGroup = 0;
            }
            matrix_dirty.clear();
        }
        
        // Emit obj_key into destination and return its traversable depth.
        // Unary groups emit their descendants directly; other groups append a
        // cached or newly built IAS instance.
        treeLook = [this, &context](const std::string& obj_key, nlohmann::json& renderGroup, decltype(nodeCache)& nodeCache,
                                   InstanceCollector& destination, const m3r4c& transform, const uint32_t instance_id, const TreeLookPolicy& policy) -> uint
        {
            const auto candidate_it = candidates.find(obj_key);
            if (candidate_it != candidates.end()) {
                const auto& candidate = candidate_it->second;
                if (candidate.handle == 0) return 0;
                appendInstance(destination, candidate.handle, transform, instance_id, candidate.sbt, candidate.vmask);
                return 1;
            }

            const auto group_it = renderGroup.find(obj_key);
            if (group_it == renderGroup.end() || group_it->empty()) return 0;

            if (group_it->size() == 1 && policy.collapseSingleChild) {

                uint8_t max_depth = 0;
                // One child key can still expand to many matrix instances.
                forEachGroupChildInstance(*group_it, destination, [&](const std::string& item_key, const m3r4c& local_transform, uint32_t child_instance_id) {
                    auto tmp_matrix = composeAffine(transform, local_transform);
                    auto test_depth = treeLook(item_key, renderGroup, nodeCache, destination, tmp_matrix, child_instance_id, policy);
                    max_depth = max(max_depth, test_depth);
                });
                return max_depth;
            }

            auto& node = nodeCache[obj_key];
            if (nullptr == node) {
                node = std::make_shared<SceneNode>();
            } else if (node->frame != UINT32_MAX && node->handle != 0) {
                appendInstance(destination, node->handle, transform, instance_id);
                return node->depth;
            }

            InstanceCollector collector(*node);
            uint8_t child_depth = 0;
            forEachGroupChildInstance(*group_it, collector, [&](const std::string& item_key, const m3r4c& local_transform, uint32_t child_instance_id) {
                auto test_depth = treeLook(item_key, renderGroup, nodeCache, collector, local_transform, child_instance_id, policy);
                child_depth = max(child_depth, test_depth);
            });

            xinxinoptix::buildIAS(context, node->hostInstances, *node,
                policy.allowUpdate && collector.canUpdate(), policy.preferFastBuild, !policy.preferFastBuild);
            collector.commit();

            if (node->handle == 0) return 0;
            node->frame = frameid;
            node->depth = node->hostInstances.empty()? 0 : child_depth + 1;

            appendInstance(destination, node->handle, transform, instance_id);
            return node->depth;
        };

        groupTask = [this, &context](std::string_view group_key, std::string_view entry_key, bool cameraSpace, decltype(nodeCache)& nodeCache, bool allow_update) -> uint64_t
        {
            if (!sceneJson.contains(group_key)) return 0;
            if (!sceneJson.contains(entry_key)) return 0;

            auto& rg = sceneJson[group_key];
            auto& entrys = sceneJson[entry_key];
            const TreeLookPolicy policy { cameraSpace && collapseDynamicSingleChildIAS, allow_update, cameraSpace };

            auto& top_node = nodeCache[group_key.data()];
            if (nullptr == top_node) top_node = std::make_shared<SceneNode>();
            InstanceCollector collector(*top_node);
            uint8_t child_depth = 0;
            uint32_t instance_id = 0;

            for (const auto& entry : entrys.items()) {
                const auto& key = entry.key();
                const auto& matrix_keys = entry.value();
                for (const auto& matrix_key_json : matrix_keys.items()) {
                    const auto& matrix_key = matrix_key_json.value().get_ref<const std::string&>();
                    const auto matrix_it = matrix_map.find(matrix_key);
                    if (matrix_it == matrix_map.end()) continue;
                    collector.reserveAdditional(matrix_it->second.size());
                    for (auto matrix : matrix_it->second) {
                        if (cameraSpace) {
                            matrix[3] -= cameraPos.x;
                            matrix[7] -= cameraPos.y;
                            matrix[11] -= cameraPos.z;
                        }
                        auto test_depth = treeLook(key, rg, nodeCache, collector, matrix, instance_id++, policy);
                        child_depth = max(child_depth, test_depth);
                    }
                }
            }

            if (top_node->hostInstances.empty()) {
                collector.commit();
                top_node->depth = 0;
                return 0;
            }

            xinxinoptix::buildIAS(context, top_node->hostInstances, *top_node, policy.allowUpdate && collector.canUpdate(), policy.preferFastBuild, !policy.preferFastBuild);
            collector.commit();
            top_node->frame = frameid;
            top_node->depth = child_depth + 1;
            maxNodeDepth = max(top_node->depth, maxNodeDepth);
            return top_node->handle;
        };

        maxNodeDepth = 1;
        {
            dynamicRenderGroup = groupTask("DynamicRenderGroups", "DynamicEntries", true, nodeCache, true);
        }
        if (0 == staticRenderGroup) {
            staticRenderGroup = groupTask("StaticRenderGroups", "StaticEntries", false, nodeCacheStatic, false);
        } else {
            uint8_t depth = 1;
            auto find = nodeCacheStatic.find("StaticRenderGroups");
            if (find != nodeCacheStatic.end()) {
                depth = find->second->depth;
            }
            maxNodeDepth = max(maxNodeDepth, depth);
        }
        maxNodeDepth += 1;
        gather();
    }

    uint8_t maxNodeDepth = 1;
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCache {};
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCacheStatic {};
    bool collapseDynamicSingleChildIAS{true};

    float3 cameraPos;
    SceneNode rootNode;
    IASBuildWorkspace iasBuildWorkspace;
    uint64_t rootLightChild {};
    uint64_t rootStaticChild {};
    uint64_t rootDynamicChild {};

    uint64_t staticRenderGroup {};
    uint64_t dynamicRenderGroup {};
    
    std::unordered_map<std::string, Candidate> candidates {};
    bool volume_scene_bindings_dirty {};

    std::function<uint8_t(const std::string&, nlohmann::json& renderGroup, decltype(nodeCache)& nodeCache, 
        InstanceCollector& destination, const m3r4c& transform, const uint32_t instance_id, const TreeLookPolicy& policy)> treeLook;

    std::function<uint64_t(std::string_view group_key, std::string_view entry_key, bool cameraSpace,
        decltype(nodeCache)& nodeCache, bool allow_update)> groupTask;

    void preload_mesh(const std::string &key, const std::string &mtlid, 
        const std::vector<std::string> &matNames, const int *matIds,
        const float *verts, size_t numverts, const uint *tris, size_t numtris,
        std::map<std::string, std::pair<float const *, size_t>> const &vtab, const GeoChange change=GeoChange::FullChange);

    void unload_object(std::string const &key) {
        if (cleanTasks.count(key)==0) return;
        cleanTasks[key](key);
        cleanTasks.erase(key);
    }

    bool consumeVolumeSceneBindingsDirty() {
        const bool dirty = volume_scene_bindings_dirty;
        volume_scene_bindings_dirty = false;
        if (dirty) candidateBindingsDirty = true;
        return dirty;
    }

    void setCollapseDynamicSingleChildIAS(bool enabled) {
        if (collapseDynamicSingleChildIAS == enabled) return;
        collapseDynamicSingleChildIAS = enabled;
        for (auto& [_, node] : nodeCache) {
            if (!node) continue;
            node->frame = UINT32_MAX;
            node->childHandles.clear();
            node->childHandlesScratch.clear();
        }
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

    void bakeVolumeDensityForCurrentFrame(const Params& params) {

        if (_vdb_grids_cached.empty()) { return; }

        const auto makeReadableVDBKey = [](const std::string& vdb_key) {
            const auto hash_pos = vdb_key.rfind('#');
            if (hash_pos == std::string::npos || hash_pos + 1 >= vdb_key.size()) {
                return vdb_key;
            }

            try {
                size_t parsed_chars = 0;
                const int type_index = std::stoi(vdb_key.substr(hash_pos + 1), &parsed_chars);
                if (parsed_chars != vdb_key.size() - hash_pos - 1) {
                    return vdb_key;
                }

                const auto element_type = static_cast<zeno::TextureObjectVDB::ElementType>(type_index);
                const auto type_name = magic_enum::enum_name(element_type);
                if (type_name.empty()) {
                    return vdb_key;
                }

                return vdb_key.substr(0, hash_pos + 1) + std::string(type_name);
            } catch (...) {
                return vdb_key;
            }
        };

        std::unordered_set<uint> volmats{};
        cookGeoMatrix(volmats);

        for (auto shader_index : volmats) {
            if (shader_index >= OptixUtil::rtMaterialShaders.size()) { continue; }

            auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];
            if (shader_ref.vbds.empty()) { continue; }
            if (!shader_ref.callable_prg || shader_ref.callable_prg->ptx.empty()) { continue; }

            const auto density_slot = shader_ref.density_vdb_primary_slot;
            if (density_slot >= shader_ref.vbds.size()) { continue; }

            const auto& density_vdb_key = shader_ref.vbds[density_slot];
            auto density_volume_it = _vdb_grids_cached.find(density_vdb_key);
            if (density_volume_it == _vdb_grids_cached.end()) { continue; }

            auto& density_volume = *density_volume_it->second;
            auto slot = std::min<uint>(density_volume.density_grid_index, density_volume.grids.size() - 1u);
            auto density_grid = density_volume.grids[slot];
            if (!density_grid) { continue; }
            const uint8_t octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(density_volume.octreeBuildDepth);

            if (!density_volume.use_gpu_baked_octree) {
                if (density_volume.baked_density && density_volume.baked_density->valid()) {
                    density_volume.baked_density->reset();
                    density_volume.density_bake_key.clear();
                    buildVolumeAccel(density_volume, OptixUtil::context);
                    volume_scene_bindings_dirty = true;
                }
                continue;
            }

            HitGroupData hit_group = {};
            const auto vdb_count = std::min<size_t>(shader_ref.vbds.size(), 8);
            for (size_t i = 0; i < vdb_count; ++i) {
                const auto& vdb_key = shader_ref.vbds[i];
                auto volume_it = _vdb_grids_cached.find(vdb_key);
                if (volume_it == _vdb_grids_cached.end()) { continue; }

                const auto& volume = volume_it->second;
                if (!volume || volume->grids.empty() || !volume->grids.front()) { continue; }
                hit_group.vdb_grids[i] = volume->grids.front()->buffer.handle;
            }

            for (uint i = 0; i < 32; ++i) {
                hit_group.textures[i] = shader_ref.getTexture(i);
            }
            if (shader_ref.parameters.contains("vol_depth")) {
                hit_group.vol_depth = shader_ref.parameters["vol_depth"];
            }
            if (shader_ref.parameters.contains("vol_extinction")) {
                hit_group.vol_extinction = shader_ref.parameters["vol_extinction"];
            }

            std::vector<std::string> compile_macros;
            const bool generated_callable = shader_ref.callable_src.find("//COMMON_CODE") == std::string::npos;
            if (generated_callable) {
                compile_macros.push_back("--define-macro=__FORWARD__");
            }
            for (const auto& [key, value] : shader_ref.macros) {
                compile_macros.push_back("--define-macro=" + key + "=" + value);
            }

            std::ostringstream key_stream;
            const bool has_density_signature = !shader_ref.density_signature.empty();
            key_stream << shader_index << ':' << density_slot << ':';
            if (has_density_signature) {
                key_stream << "density signature=" << shader_ref.density_signature << ';';
            } else {
                key_stream << std::hash<std::string>{}(shader_ref.callable_src) << ':'
                    << shader_ref.callable_src.size() << ':';
                for (const auto& macro : compile_macros) {
                    key_stream << macro << ';';
                }
            }
            key_stream << "octreeBuildDepth=" << unsigned(octreeBuildDepth) << ';';
            if (density_volume.use_custom_density_sample_bbox) {
                key_stream << "sampleBBox="
                    << density_volume.custom_density_sample_min.x << ','
                    << density_volume.custom_density_sample_min.y << ','
                    << density_volume.custom_density_sample_min.z << ':'
                    << density_volume.custom_density_sample_max.x << ','
                    << density_volume.custom_density_sample_max.y << ','
                    << density_volume.custom_density_sample_max.z << ';';
            }
            if (has_density_signature) {
                for (const size_t slot : shader_ref.density_vdb_referenced_slots) {
                    if (slot < shader_ref.vbds.size()) {
                        key_stream << "vdb" << slot << '=' << shader_ref.vbds[slot] << ';';
                    }
                }
            } else {
                key_stream << "params=" << shader_ref.parameters.dump() << ';';
                for (const auto& vdb_key : shader_ref.vbds) {
                    key_stream << "vdb=" << vdb_key << ';';
                }
                for (uint i = 0; i < 32; ++i) {
                    key_stream << "tex" << i << '=' << static_cast<uint64_t>(hit_group.textures[i]) << ';';
                }
            }
            key_stream << "density=" << density_vdb_key;
            const std::string bake_key = key_stream.str();

            const bool force_density_bake = shader_ref.force_density_bake;
            shader_ref.force_density_bake = false;

            if (!force_density_bake && density_volume.density_bake_key == bake_key) {
                continue;
            }

            std::ostringstream label_stream;
            label_stream << "shader=" << shader_index
                << " density_slot=" << density_slot
                << " " << makeReadableVDBKey(density_vdb_key);
            const std::string label = label_stream.str();

            xinxinoptix::VolumeDensityBakeInputs bake_inputs;
            bake_inputs.params = &params;
            bake_inputs.hit_group = &hit_group;
            bake_inputs.callable_ptx = &shader_ref.callable_prg->ptx;
            bake_inputs.callable_module_key = &shader_ref.callable_prg->cache_key;
            bake_inputs.callable_source = shader_ref.callable_src.c_str();
            bake_inputs.density_signature = shader_ref.density_signature.c_str();
            bake_inputs.validation_label = label.c_str();
            bake_inputs.seed = 0x12345678u ^ static_cast<uint32_t>(shader_index * 1664525u + density_slot);

            xinxinoptix::VolumeDensityBakeResult bake_result;
            xinxinoptix::VolumeDensityBakeOptions bake_options;
            bake_options.octreeBuildDepth = octreeBuildDepth;
            bake_options.validate_sparse_octree = density_volume.validate_gpu_baked_octree;
            bake_options.use_custom_sample_bbox = density_volume.use_custom_density_sample_bbox;
            bake_options.custom_sample_min = density_volume.custom_density_sample_min;
            bake_options.custom_sample_max = density_volume.custom_density_sample_max;
            const bool baked = density_volume.baked_density
                && bakeDensityGridToSparseBricksOnGPU(*density_grid, *density_volume.baked_density, bake_inputs, bake_options, &bake_result);
            if (baked) {
                density_volume.density_bake_key = bake_key;
                buildVolumeAccel(density_volume, OptixUtil::context);
                volume_scene_bindings_dirty = true;
                std::cout << "Volume density sparse bake {" << label << "} max=" << bake_result.max_density << std::endl;
            } else {
                density_volume.density_bake_key.clear();
                if (density_volume.baked_density) {
                    density_volume.baked_density->reset();
                }
                std::cerr << "Volume density bake failed {" << label << "}" << std::endl;
            }
        }
    }

    void prepareVolumeAssets() {

        if (_vdb_grids_cached.empty()) { return; }

        std::unordered_set<uint> volmats{};
        cookGeoMatrix(volmats);

        std::unordered_set<std::string> required {};
        std::unordered_set<std::string> accel_required {};

        for(auto shader_index : volmats) {
            if (shader_index >= OptixUtil::rtMaterialShaders.size()) { continue; }

            const auto& shader_ref = OptixUtil::rtMaterialShaders[shader_index];
            if ( shader_ref.vbds.size() == 0 ) { continue; }

            for (const auto& vdb_key : shader_ref.vbds) {
                required.insert(vdb_key);
            }    
            const auto density_slot = shader_ref.density_vdb_primary_slot;
            if (density_slot < shader_ref.vbds.size()) {
                accel_required.insert(shader_ref.vbds[density_slot]);
            }
        }

        for (auto const& [key, vol] : _vdb_grids_cached) {

            if (!required.count(key)) {
                if (vol->node->handle != 0) {
                    volume_scene_bindings_dirty = true;
                }
                releaseVolumeDeviceData(*vol);
                continue;
            }

            const bool needs_accel = accel_required.count(key) != 0;

            if (vol->dirty) {
                for (auto& task : vol->tasks) {
                    task();
                }
            }

            if (needs_accel) {
                if (vol->dirty || vol->node->handle == 0) {
                    if (!vol->grids.empty() && !vol->aggregate.octree.empty()) {
                        buildVolumeAccel(*vol, OptixUtil::context);
                        volume_scene_bindings_dirty = true;
                    } else {
                        if (vol->node->handle != 0) {
                            volume_scene_bindings_dirty = true;
                        }
                        cleanupVolumeAccel(*vol->node);
                    }
                }
            } else {
                if (vol->node->handle != 0) {
                    volume_scene_bindings_dirty = true;
                }
                cleanupVolumeAccel(*vol->node);
            }

            vol->dirty = false;
        }
    }

    bool preloadVolumeBox(const std::string& key, std::string& matid, uint8_t bounds, glm::mat4& transform, std::vector<sutil::Aabb>& aabbs);
    bool preloadVDB(const zeno::TextureObjectVDB& texVDB, std::string& combined_key);
};


inline OptixScene defaultScene;