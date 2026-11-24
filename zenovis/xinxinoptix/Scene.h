#pragma once

#include <set>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
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
#include "ClusterRuntime.h"
#include "ClusterMaterials.h"
#include "WaterScene.h"
#include "ScenePlacements.h"
#include "SceneDependencies.h"

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
    std::optional<std::string> pendingSceneSource;
    struct PendingMatrices {std::vector<m3r4c> matrices;std::vector<int> ids;};
    std::map<std::string,PendingMatrices> pendingMatrices;
    static std::mutex& registrationMutex() {static std::mutex mutex;return mutex;}
    bool candidateBindingsDirty {true};
    bool sceneGraphDirty {true};
    zeno::scene::Dependencies sceneDependencies;
    struct GeometryPublication {
        OptixTraversableHandle handle=0;
        uint64_t version=0;
        uint8_t depth=0;
    };
    std::map<std::string,GeometryPublication> streamedGeometry;
    std::set<std::pair<bool,std::string>> mutableGroups;
    struct OwnedGroup {SceneNode* node;std::string name;bool isStatic;};
    std::unordered_map<OptixTraversableHandle,OwnedGroup> ownedGroupHandles;
    SceneNode* fusedRootGroup=nullptr;
    std::string fusedRootName;
    bool fusedRootStatic=false;
    bool skipFusedRootBuild=false,sourceTopologyDirty=true;
    bool rootBoundsDirty=true;
    std::optional<m3r4c> rootCamera;

    void refreshMutableGroups() {
        mutableGroups.clear();
        for(const auto& [name,_]:streamedGeometry)
            for(const auto& group:sceneDependencies.ancestors(name))mutableGroups.emplace(group.isStatic,group.name);
    }
    void publishGeometry(const std::string& name,OptixTraversableHandle handle,uint64_t version,uint8_t depth) {
        auto inserted=streamedGeometry.try_emplace(name);
        auto& previous=inserted.first->second;
        if(!inserted.second && previous.handle==handle && previous.version==version && previous.depth==depth)return;
        sourceTopologyDirty|=inserted.second || bool(previous.handle)!=bool(handle) || previous.depth!=depth;
        previous={handle,version,depth};
        sceneDependencies.geometryChanged(name);rootBoundsDirty=true;
        if(inserted.second)refreshMutableGroups();
    }
    void removePublishedGeometry(const std::string& name) {
        if(!streamedGeometry.erase(name))return;
        sourceTopologyDirty=true;
        sceneDependencies.geometryChanged(name);rootBoundsDirty=true;refreshMutableGroups();
    }

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
    bool lightIasDirty = false;
    
public:
    std::shared_ptr<zeno::cluster::Runtime> clusterAsset;
    std::unique_ptr<zeno::cluster::Materials> clusterMaterials;
    std::string clusterAssetKey, clusterAssetManifest, clusterAssetRevision;
    zeno::cluster::RuntimeSettings clusterSettings;
    bool clusterRootDirty = false;
    bool clusterUnloadPending = false;
    std::string clusterObjectName;
    std::vector<m3r4c> clusterPlacements;
    std::map<std::string,zeno::water::Settings> waterSettings;
    std::map<std::string,std::string> waterMaterialNames;
    std::map<std::string,std::string> waterObjectNames;
    std::map<std::string,std::vector<m3r4c>> waterPlacements;
    zeno::mega::PlacementInvalidation placementsDirty;
    bool waterVisible=false;
    bool streamPlacementsChanged=false;
    std::set<std::string> waterUnloadPending;
    std::unique_ptr<zeno::water::SceneRuntime> water;
    bool waterRootDirty=false;
    void preloadWater(const std::string& key,const Json& json,const std::string& material,const std::string& objectName) {
        zeno::water::Settings s;
        s.center=json.at("center").get<std::array<float,3>>();s.size=json.at("size").get<std::array<float,2>>();
        s.wind=json.at("wind").get<std::array<float,2>>();
        s.amplitude=json.at("amplitude");s.period=json.at("period");s.windSpeed=json.at("windSpeed");s.depth=json.at("depth");
        s.time=json.at("time");s.edgePixels=json.at("edgePixels");
        s.loopDuration=json.value("loopDuration",0.f);
        s.boundaryWidth=json.at("boundaryWidth");s.resolution=json.at("resolution");s.seed=json.at("seed");
        s.maxClusters=json.at("maxClusters");s.ellipse=json.at("ellipse");zeno::water::validate(s);
        // Graphics constructors can register several water nodes in parallel.
        static std::mutex registrationMutex;
        const std::lock_guard<std::mutex> registration(registrationMutex);
        if(waterObjectNames.find(key)==waterObjectNames.end() || waterObjectNames.at(key)!=objectName)placementsDirty=true;
        waterObjectNames[key]=objectName;
        waterSettings[key]=s;waterMaterialNames[key]=material;waterUnloadPending.erase(key);
        maxNodeDepth=max(maxNodeDepth,decltype(maxNodeDepth)(3));
    }
    void finalizeWaterRegistration() {
        // Descriptor keys can change on a new graph frame while the named GPU
        // resource survives. Retire publications only after resolving all keys.
        for(const auto& key:waterUnloadPending){waterSettings.erase(key);waterMaterialNames.erase(key);waterObjectNames.erase(key);placementsDirty=true;}
        waterUnloadPending.clear();
        std::vector<std::string> removed;
        for(const auto& [name,_]:streamedGeometry)if(name!=clusterObjectName &&
            std::none_of(waterObjectNames.begin(),waterObjectNames.end(),[&](const auto& item){return item.second==name;}))removed.push_back(name);
        for(const auto& name:removed)removePublishedGeometry(name);
        if(waterSettings.empty() && water){water.reset();waterRootDirty=true;waterVisible=false;}
    }
    bool isStreamedGeometry(const std::string& name) const {
        if(clusterAsset && name==clusterObjectName)return true;
        for(const auto& [_,waterName]:waterObjectNames)if(waterName==name)return true;
        return false;
    }
    bool streamedGeometryInScene(const std::string& name) const {return sceneDependencies.references(name);}
    void clearScenePublications() {
        streamedGeometry.clear();mutableGroups.clear();rootCamera.reset();rootBoundsDirty=true;
        fusedRootGroup=nullptr;fusedRootName.clear();sourceTopologyDirty=true;
        // A renderer can be destroyed while the unchanged logical scene remains.
        sceneDependencies.compile(sceneJson);
        for(auto* cache:{&nodeCache,&nodeCacheStatic})for(auto& [_,node]:*cache)if(node)node->valid=false;
    }
    bool isStreamedBinding(const std::string& key) const {
        if(!sceneJson.contains(brikey) || !sceneJson.at(brikey).contains(key))return false;
        return isStreamedGeometry(sceneJson.at(brikey).at(key).value("Geom",std::string{}));
    }
    void resolveStreamPlacements() {
        finalizeSceneRegistration();
        if(!placementsDirty.exchange(false))return;
        std::set<std::string> names;
        if(clusterAsset)names.insert(clusterObjectName);
        for(const auto& [_,name]:waterObjectNames)names.insert(name);
        const auto resolved=zeno::mega::resolvePlacements(sceneJson,names,[&](const std::string& key)->const std::vector<m3r4c>* {
            const auto found=matrix_map.find(key);return found==matrix_map.end()?nullptr:&found->second;
        });
        const auto next=clusterAsset?resolved.at(clusterObjectName).world:std::vector<m3r4c>{};
        if(next!=clusterPlacements){clusterPlacements=next;clusterRootDirty=true;streamPlacementsChanged=true;}
        std::map<std::string,std::vector<m3r4c>> nextWater;
        for(const auto& [key,name]:waterObjectNames)nextWater[key]=resolved.at(name).world;
        if(nextWater!=waterPlacements){waterPlacements=std::move(nextWater);streamPlacementsChanged=true;}
    }
    const std::vector<m3r4c>& clusterScenePlacements() {resolveStreamPlacements();return clusterPlacements;}
    bool consumeStreamPlacementChanges() {const auto changed=streamPlacementsChanged;streamPlacementsChanged=false;return changed;}
    bool updateWater(const Params& params,bool interactive) {
        if(waterSettings.empty())return false;
        resolveStreamPlacements();
        std::vector<zeno::water::SceneSurface> settings;settings.reserve(waterSettings.size());
        uint32_t maxSbt=0;
        for(const auto& [key,surface]:waterSettings) {
            auto s=surface;
            s.sbt=shader_indice_table.at(std::make_tuple(zeno::water::shaderKey(waterMaterialNames.at(key)),ShaderMark::Mesh));
            maxSbt=std::max(maxSbt,s.sbt);
            const auto& placements=waterPlacements.at(key);
            s.poolToWorld=placements.empty()?IdentityMatrix:placements.front();
            if(placements.size()>1)s.instanceTransforms=placements;
            settings.push_back({waterObjectNames.at(key),std::move(s),!placements.empty()});
        }
        if(!water)water=std::make_unique<zeno::water::SceneRuntime>();
        const float projection=params.physical_camera_panorama_camera?float(params.height)/3.14159265358979323846f:
            float(params.height)*params.cam.focal_length/std::max(params.cam.height,.001f);
        const auto& camera=params.cam;
        zeno::water::View view{{camera.eye.x,camera.eye.y,camera.eye.z},projection};
        view.right={camera.right.x,camera.right.y,camera.right.z};view.up={camera.up.x,camera.up.y,camera.up.z};view.front={camera.front.x,camera.front.y,camera.front.z};
        if(!params.physical_camera_panorama_camera && camera.pitch==0.f && camera.yaw==0.f) {
            view.tanX=camera.width/(2.f*camera.focal_length)*(1.f+2.f*std::abs(camera.horizontal_shift));
            view.tanY=camera.height/(2.f*camera.focal_length)*(1.f+2.f*std::abs(camera.vertical_shift));
        }
        const bool changed=water->update(OptixUtil::context,maxSbt,settings,view,interactive);
        waterVisible=false;
        for(const auto& source:water->publications()) {
            publishGeometry(source.key,source.handle,source.version,1);
            waterVisible|=source.handle!=0;
        }
        waterRootDirty|=changed;return changed;
    }
    bool canUseOpaqueClusterShadows() const {
        // Legacy geometry can require node-graph opacity, custom intersections
        // or OMM handling. Keep its established enforced-any-hit trace path.
        // Structured water keeps any-hit enabled in its grid template; its
        // bound callable determines opacity and shadow transmission.
        return geoTypeMap.empty() && shader_indice_table.count(
            std::make_tuple(std::string(zeno::cluster::MaterialShaderName),ShaderMark::Mesh));
    }
    void preloadClusterAsset(const std::string& key,const std::string& path,const std::string& revision,
                             zeno::cluster::RuntimeSettings settings,const std::string& objectName) {
        if(clusterObjectName!=objectName){removePublishedGeometry(clusterObjectName);clusterObjectName=objectName;placementsDirty=true;}
        if(clusterAsset && clusterAssetManifest==path && clusterAssetRevision==revision &&
           clusterSettings.geometryBytes==settings.geometryBytes && clusterSettings.clasBytes==settings.clasBytes &&
           clusterSettings.pixelError==settings.pixelError && clusterSettings.loadsPerFrame==settings.loadsPerFrame &&
           clusterSettings.materials==settings.materials && clusterSettings.textureMiB==settings.textureMiB &&
           clusterSettings.hostTextureMiB==settings.hostTextureMiB) {
            clusterAssetKey=key;clusterUnloadPending=false;return;
        }
        auto next=std::make_shared<zeno::cluster::Runtime>(std::filesystem::u8path(path),settings);
        clusterMaterials.reset();clusterAsset=std::move(next);
        removePublishedGeometry(clusterObjectName);
        clusterAssetKey=key;clusterAssetManifest=path;clusterAssetRevision=revision;clusterSettings=settings;
        clusterUnloadPending=false;
        clusterRootDirty=true;
        placementsDirty=true;
        maxNodeDepth=max(maxNodeDepth,decltype(maxNodeDepth)(3));
    }
    void finalizeClusterAssetRegistration() {
        // GraphicsManager replaces frame descriptors in two insert passes.
        // Delay deletion until both passes finish so a refreshed reference can
        // retain its renderer-owned residency instead of reloading every frame.
        if(!clusterUnloadPending)return;
        removePublishedGeometry(clusterObjectName);
        clusterMaterials.reset();clusterAsset.reset();clusterAssetKey.clear();clusterAssetManifest.clear();clusterAssetRevision.clear();
        clusterObjectName.clear();placementsDirty=true;
        clusterRootDirty=true;clusterUnloadPending=false;
    }
    bool updateClusters(const Params& params,bool interactive,zeno::cluster::LodControl control = {}) {
        if(!clusterAsset)return false;
        resolveStreamPlacements();
        if(clusterPlacements.empty())return false;
        if(clusterSettings.materials && !clusterMaterials) {
            clusterMaterials=std::make_unique<zeno::cluster::Materials>(std::filesystem::u8path(clusterAssetManifest),clusterSettings.textureMiB,clusterSettings.hostTextureMiB);
            clusterAsset->setOpaqueShadowPolicy(clusterMaterials->opaqueShadowPolicy());
        }
        clusterAsset->setHostCacheFillAllowed(!clusterMaterials||!clusterMaterials->primaryDetailPending());
        auto shader=shader_indice_table.find(std::make_tuple(std::string(zeno::cluster::MaterialShaderName),ShaderMark::Mesh));
        const auto index=shader==shader_indice_table.end()?
            shader_indice_table.at(std::make_tuple(std::string(""),ShaderMark::Mesh)):shader->second;
        const float projection=params.physical_camera_panorama_camera?float(params.height)/3.14159265358979323846f:
            float(params.height)*params.cam.focal_length/std::max(params.cam.height,0.001f);
        const float right[]{params.cam.right.x,params.cam.right.y,params.cam.right.z};
        const float up[]{params.cam.up.x,params.cam.up.y,params.cam.up.z};
        const float forward[]{params.cam.front.x,params.cam.front.y,params.cam.front.z};
        // Extra detail from a lens narrower than 60 degrees follows the image.
        // Periphery keeps the existing wide-lens metric, including secondary rays.
        const float peripheralProjection=float(params.height)*0.866025404f;
        const auto zoom=params.physical_camera_panorama_camera?zeno::cluster::makePanoramaView(up,float(params.width),float(params.height),
                params.physical_camera_panorama_vr180?params.physical_camera_pupillary_distance*.5f:0.f):
            projection<=peripheralProjection ? zeno::cluster::ZoomView{} : zeno::cluster::makeZoomView(right,up,forward,
            params.cam.width/(2.f*params.cam.focal_length),params.cam.height/(2.f*params.cam.focal_length),
            peripheralProjection,params.cam.horizontal_shift,params.cam.vertical_shift);
        if (const char* timing = std::getenv("ZENO_OPTIX_RENDER_TIMING"); timing && timing[0] == '1')
            zeno::log_info("CLAS view: eye={},{},{}, projection={}",
                params.cam.eye.x, params.cam.eye.y, params.cam.eye.z, projection);
        zeno::cluster::LodCamera base{{params.cam.eye.x,params.cam.eye.y,params.cam.eye.z},projection,zoom};
        std::vector<zeno::cluster::LodCamera> views;views.reserve(clusterPlacements.size());
        for(const auto& world:clusterPlacements)views.push_back(cameraInAsset(world,base));
        const auto& primary=views.front();
        const bool changed=clusterAsset->update(OptixUtil::context,index,
            {primary.eye[0],primary.eye[1],primary.eye[2]},primary.projection,interactive,primary.zoom,control,
            views.size()>1?views:std::vector<zeno::cluster::LodCamera>{});
        publishGeometry(clusterObjectName,clusterAsset->handle(),clusterAsset->stats().sceneVersion,2);
        clusterRootDirty|=changed;return changed;
    }
    static zeno::cluster::LodCamera cameraInAsset(const m3r4c& world,const zeno::cluster::LodCamera& base) {
        if(world==IdentityMatrix)return base;
        const auto inverse=zeno::mega::inverse(world);
        auto camera=base;const auto eye=zeno::mega::point(inverse,{base.eye[0],base.eye[1],base.eye[2]});
        std::copy(eye.begin(),eye.end(),camera.eye);
        const auto norm=[](const m3r4c& m) {
            float largest=0;
            for(unsigned i=0;i<3;++i) {
                float row=0;for(unsigned j=0;j<3;++j) {
                    float value=0;for(unsigned k=0;k<3;++k)value+=m[k*4+i]*m[k*4+j];
                    row+=std::abs(value);
                }
                largest=std::max(largest,row);
            }
            return std::sqrt(largest);
        };
        const float inverseScale=norm(inverse),condition=std::max(1.f,norm(world)*inverseScale);
        camera.projection*=condition;
        if(base.zoom.panoramaVertical>0.f) {
            float length2=0;
            for(unsigned i=0;i<3;++i) {
                camera.zoom.panoramaUp[i]=0;
                for(unsigned j=0;j<3;++j)camera.zoom.panoramaUp[i]+=inverse[i*4+j]*base.zoom.panoramaUp[j];
                length2+=camera.zoom.panoramaUp[i]*camera.zoom.panoramaUp[i];
            }
            for(float& v:camera.zoom.panoramaUp)v/=std::sqrt(length2);
            camera.zoom.panoramaVertical*=condition*condition;camera.zoom.panoramaHorizontal*=condition*condition;
            camera.zoom.stereoRadius*=inverseScale;
        } else if(condition>1.0001f) {
            // A conservative omnidirectional metric avoids under-refinement
            // when a nonuniform parent transform distorts the viewing cone.
            camera.zoom={};
        } else if(base.zoom.peripheralProjection>0.f) {
            for(unsigned p=0;p<4;++p) {
                float length2=0;
                for(unsigned i=0;i<3;++i) {
                    camera.zoom.planes[p][i]=0;
                    for(unsigned j=0;j<3;++j)camera.zoom.planes[p][i]+=world[j*4+i]*base.zoom.planes[p][j];
                    length2+=camera.zoom.planes[p][i]*camera.zoom.planes[p][i];
                }
                for(float& v:camera.zoom.planes[p])v/=std::sqrt(length2);
            }
        }
        return camera;
    }
    uint32_t frameid;
    phmap::parallel_flat_hash_map_m<std::string, std::pair<glm::vec3, glm::vec3>> mesh_bbox;
    phmap::parallel_flat_hash_map_m<std::string, std::vector<glm::mat4>> glm_matrix_map;
    phmap::parallel_node_hash_map_m<std::string, std::shared_ptr<VolumeWrapper>> _vdb_grids_cached;

    bool shaderIndexMatches(const shader_key_t& key,size_t index) const {
        const auto current=shader_indice_table.find(key);
        return current!=shader_indice_table.end() && current->second==index;
    }
    inline void load_shader_indice_table(std::unordered_map<shader_key_t, uint16_t, ByShaderKey> &table) {
        const bool unchanged = table.size() == shader_indice_table.size()
            && std::all_of(table.begin(), table.end(), [&](const auto& item) {
                const auto current = shader_indice_table.find(item.first);
                return current != shader_indice_table.end() && current->second == item.second;
            });
        if (unchanged) return;

        const auto lightIndex = [](const auto& indices, ShaderMark mark) {
            const auto it = indices.find(std::tuple{std::string("Light"), mark});
            return it == indices.end() ? UINT16_MAX : it->second;
        };
        // The light GAS can stay identical while a new material changes the
        // light instance's SBT offset. That still requires a light IAS update.
        lightIasDirty |= lightIndex(shader_indice_table, ShaderMark::Mesh) != lightIndex(table, ShaderMark::Mesh)
            || lightIndex(shader_indice_table, ShaderMark::Sphere) != lightIndex(table, ShaderMark::Sphere);

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
        const std::lock_guard<std::mutex> lock(registrationMutex());
        pendingMatrices[std::move(key)]={std::move(matrix_list),std::move(instance_ids)};
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
        const std::lock_guard<std::mutex> lock(registrationMutex());
        pendingSceneSource=jsonString;
    }

    void finalizeSceneRegistration() {
        // Both GraphicsManager insertion passes finish before this boundary.
        // Intermediate empty descriptors must not invalidate the retained AS.
        const std::lock_guard<std::mutex> lock(registrationMutex());
        if(pendingSceneSource) {
            auto source=std::move(*pendingSceneSource);pendingSceneSource.reset();
            if(!sceneJsonSource || *sceneJsonSource!=source)try {
                auto next=nlohmann::json::parse(source);
                if(next!=sceneJson) {
                    zeno::scene::Dependencies dependencies;dependencies.compile(next);
                    sceneJson=std::move(next);sceneDependencies=std::move(dependencies);
                    refreshMutableGroups();placementsDirty=true;
                    candidateBindingsDirty=true;sceneGraphDirty=true;
                }
                sceneJsonSource=std::move(source);
            }catch(const std::exception& error){zeno::log_error("Invalid render scene: {}",error.what());}
        }
        for(auto& [key,next]:pendingMatrices) {
            const auto previous=matrix_map.find(key);const auto ids=instance_ids_map.find(key);
            if(previous!=matrix_map.end() && previous->second==next.matrices &&
                (next.ids.empty()?ids==instance_ids_map.end():ids!=instance_ids_map.end() && ids->second==next.ids))continue;
            matrix_map[key]=std::move(next.matrices);
            if(next.ids.empty())instance_ids_map.erase(key);else instance_ids_map[key]=std::move(next.ids);
            matrix_dirty.insert(key);placementsDirty=true;
        }
        pendingMatrices.clear();
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

    void markLightIASDirty() { lightIasDirty = true; }
    bool hasDirtyLightIAS() const { return lightIasDirty; }

    void prepare_light_ias(OptixDeviceContext& context) {

        // Light/Mesh and Light/Sphere already have separate SBT records.
        // Switching shape replaces the relevant GAS instance and selects its
        // existing record via sbtOffset; no SBT upload is needed.
        std::vector<OptixInstance> optix_instances;
        optix_instances.reserve(3);

        if (lightsWrapper.lightTrianglesGas != 0)
        {
            OptixInstance opinstance {};

            auto combinedID = std::tuple(std::string("Light"), ShaderMark::Mesh);
            // A missing light hit group must fail here, not silently bind SBT 0.
            auto shader_index = shader_indice_table.at(combinedID);

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
            auto shader_index = shader_indice_table.at(combinedID);

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
            auto shader_index = shader_indice_table.at(combinedID);

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
        if(!zeno::mega::visible(transform))return;
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

    void refreshCandidateBindings(const std::unordered_set<std::string>& geometry_dirty, std::unordered_set<std::string>& candidate_dirty) {
        const auto& bri = sceneJson[brikey];

        for (auto candidate_it = candidates.begin(); candidate_it != candidates.end();) {
            if (!bri.contains(candidate_it->first) || isStreamedBinding(candidate_it->first)) {
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
            if(isStreamedGeometry(geometry_name))continue;

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

    // Retained group nodes stay in scene coordinates. Only the root carries
    // the floating-origin camera translation.
    uint8_t emitSceneNode(OptixDeviceContext& context,const std::string& key,Json& groups,
                         std::unordered_map<std::string,std::shared_ptr<SceneNode>>& cache,
                         InstanceCollector& destination,const m3r4c& transform,uint32_t instanceId,bool isStatic) {
        if(sceneJson.contains(brikey) && sceneJson.at(brikey).contains(key)) {
            const auto name=sceneJson.at(brikey).at(key).value("Geom",std::string{});
            const auto source=streamedGeometry.find(name);
            if(source!=streamedGeometry.end()) {
                if(!source->second.handle)return 0;
                appendInstance(destination,source->second.handle,transform,instanceId);
                return source->second.depth;
            }
            if(isStreamedGeometry(name))return 0; // Its first complete publication is not ready.
        }
        const auto candidate=candidates.find(key);
        if(candidate!=candidates.end()) {
            const auto& c=candidate->second;if(!c.handle)return 0;
            appendInstance(destination,c.handle,transform,instanceId,c.sbt,c.vmask);return 1;
        }
        const auto group=groups.find(key);
        if(group==groups.end())return 0;
        auto& node=cache[key];if(!node)node=std::make_shared<SceneNode>();
        if(!node->valid)buildSceneGroup(context,key,*group,groups,cache,*node,isStatic);
        if(!node->depth)return 0;
        if(node->singleChild) {
            auto instance=*node->singleChild;m3r4c local;
            memcpy(local.data(),instance.transform,sizeof(instance.transform));
            const auto world=composeAffine(transform,local);
            memcpy(instance.transform,world.data(),sizeof(instance.transform));
            if(zeno::mega::visible(world))destination.append(instance);
        } else appendInstance(destination,node->handle,transform,instanceId);
        return node->depth;
    }

    void buildSceneGroup(OptixDeviceContext& context,const std::string& key,const Json& children,Json& groups,
                         std::unordered_map<std::string,std::shared_ptr<SceneNode>>& cache,SceneNode& node,bool isStatic) {
        InstanceCollector collector(node);uint8_t childDepth=0;
        forEachGroupChildInstance(children,collector,[&](const std::string& child,const m3r4c& matrix,uint32_t id) {
            childDepth=std::max(childDepth,emitSceneNode(context,child,groups,cache,collector,matrix,id,isStatic));
        });
        node.singleChild.reset();
        if(collapseDynamicSingleChildIAS && node.hostInstances.size()==1) {
            node.singleChild=node.hostInstances.front();node.depth=childDepth;
        } else if(skipFusedRootBuild && &node==fusedRootGroup) {
            // This local instance layout is copied into root with its outer
            // transform. Keep its world-space GPU cache lazy until it is needed
            // as an actual child again (for example, after adding a root light).
            node.gpuDirty=true;node.depth=node.hostInstances.empty()?0:childDepth+1;
        } else {
            // Logical static placement does not imply immutable acceleration
            // bounds. Streamed descendants make their ancestors updateable.
            const bool mutableAS=!isStatic || mutableGroups.count({isStatic,key})!=0;
            const auto previousHandle=node.handle;
            xinxinoptix::buildIAS(context,node.hostInstances,node,mutableAS && !node.gpuDirty && collector.canUpdate(),mutableAS,!mutableAS);
            node.gpuDirty=false;
            if(previousHandle!=node.handle)ownedGroupHandles.erase(previousHandle);
            if(node.handle && !ownedGroupHandles.count(node.handle))ownedGroupHandles.emplace(node.handle,OwnedGroup{&node,key,isStatic});
            node.depth=node.hostInstances.empty()?0:childDepth+1;
        }
        collector.commit();node.frame=frameid;node.valid=true;
    }

    uint64_t sceneGroup(OptixDeviceContext& context,bool isStatic) {
        const std::string category=isStatic?"StaticRenderGroups":"DynamicRenderGroups";
        const std::string entries=isStatic?"StaticEntries":"DynamicEntries";
        auto& rootInstance=isStatic?staticRootInstance:dynamicRootInstance;
        rootInstance.reset();
        if(!sceneJson.contains(category) || !sceneJson.contains(entries))return 0;
        auto& cache=isStatic?nodeCacheStatic:nodeCache;
        auto& top=cache[category];if(!top)top=std::make_shared<SceneNode>();
        if(!top->valid)
            buildSceneGroup(context,category,sceneJson.at(entries),sceneJson.at(category),cache,*top,isStatic);
        if(!top->depth)return 0;
        rootInstance=top->singleChild;
        return rootInstance?rootInstance->traversableHandle:top->handle;
    }

    inline void make_scene(OptixDeviceContext& context,float3 cam,bool camera_only) {
        finalizeSceneRegistration();
        cameraPos=cam;
        if(lightIasDirty) {prepare_light_ias(context);lightIasDirty=false;rootBoundsDirty=true;}
        if(!camera_only && sceneJson.contains(brikey)) {
            std::unordered_set<std::string> geometryDirty,candidateDirty;
            for(auto& [key,task]:dirtyTasks) {
                geometryDirty.insert(key);const auto handle=task(context);
                gas_handles[key]=handle;gas_to_obj_id[handle]=key;
            }
            dirtyTasks.clear();
            for(const auto& key:dirtyTextures) {
                const auto& texture=OptixUtil::tex_lut.at(key);
                if(texture.use_count()==1)OptixUtil::removeTexture(key);
            }
            dirtyTextures.clear();
            if(candidateBindingsDirty || !geometryDirty.empty() || !cleanTasks.empty())
                refreshCandidateBindings(geometryDirty,candidateDirty);
            for(auto& [key,task]:cleanTasks){task(key);gas_handles.erase(key);}
            cleanTasks.clear();
            for(const auto& key:candidateDirty)sceneDependencies.bindingChanged(key);
        }
        const unsigned categories=unsigned(sceneJson.contains("StaticEntries") && !sceneJson.at("StaticEntries").empty())+
            unsigned(sceneJson.contains("DynamicEntries") && !sceneJson.at("DynamicEntries").empty());
        const bool canFuseRoot=collapseDynamicSingleChildIAS && !lightsWrapper.lightIasHandle && categories==1 &&
            std::none_of(streamedGeometry.begin(),streamedGeometry.end(),[&](const auto& item){return item.second.handle && !streamedGeometryInScene(item.first);});
        // A topology/visibility/matrix edit can change which group supplies the
        // root. Make every lazy GPU cache current before changing that plan.
        skipFusedRootBuild=canFuseRoot && fusedRootGroup && !sceneGraphDirty && matrix_dirty.empty() &&
            candidates.empty() && !sourceTopologyDirty;
        if(!skipFusedRootBuild && fusedRootGroup && fusedRootGroup->gpuDirty)
            sceneDependencies.groupChanged(fusedRootName,fusedRootStatic);
        if(sceneGraphDirty) {
            fusedRootGroup=nullptr;
            // Structural edits are rare. Resource/matrix changes below only
            // touch their reverse dependencies, including shared ancestors.
            for(auto* cache:{&nodeCache,&nodeCacheStatic})
                for(auto& [_,node]:*cache)if(node)node->valid=false;
            sceneGraphDirty=false;rootBoundsDirty=true;
        }
        for(const auto& key:matrix_dirty)sceneDependencies.matrixChanged(key);
        matrix_dirty.clear();
        for(const auto& group:sceneDependencies.takeDirty()) {
            auto& cache=group.isStatic?nodeCacheStatic:nodeCache;
            const auto node=cache.find(group.name);
            if(node!=cache.end() && node->second)node->second->valid=false;
            rootBoundsDirty=true;
        }
        resolveStreamPlacements();
        staticRenderGroup=sceneGroup(context,true);
        dynamicRenderGroup=sceneGroup(context,false);
        const auto singleRoot=staticRenderGroup?staticRenderGroup:dynamicRenderGroup;
        const auto fused=canFuseRoot?ownedGroupHandles.find(singleRoot):ownedGroupHandles.end();
        fusedRootGroup=fused!=ownedGroupHandles.end()?fused->second.node:nullptr;
        fusedRootName=fusedRootGroup?fused->second.name:std::string{};
        fusedRootStatic=fusedRootGroup && fused->second.isStatic;
        sourceTopologyDirty=false;
        maxNodeDepth=1;
        for(const auto& entry:std::array<std::pair<const char*,const decltype(nodeCache)*>,2>{{
                {"StaticRenderGroups",&nodeCacheStatic},{"DynamicRenderGroups",&nodeCache}}}) {
            if(entry.second==&nodeCacheStatic?!staticRenderGroup:!dynamicRenderGroup)continue;
            const auto n=entry.second->find(entry.first);
            if(n!=entry.second->end() && n->second)maxNodeDepth=std::max(maxNodeDepth,uint8_t(1+n->second->depth));
        }
        auto camera=IdentityMatrix;camera[3]=-cam.x;camera[7]=-cam.y;camera[11]=-cam.z;
        if(lightsWrapper.lightIasHandle)maxNodeDepth=std::max(maxNodeDepth,uint8_t(3));
        for(const auto& [name,source]:streamedGeometry)if(source.handle && !streamedGeometryInScene(name))
            maxNodeDepth=std::max(maxNodeDepth,uint8_t(source.depth+1));
        if(fusedRootGroup)maxNodeDepth=fusedRootGroup->depth;
        rootBoundsDirty|=clusterRootDirty || waterRootDirty;
        if(!rootBoundsDirty && rootCamera && *rootCamera==camera)return;
        InstanceCollectorT<CudaPinnedAllocator> collector(rootNode);
        collector.reserveAdditional(4+clusterPlacements.size()+waterSettings.size());
        auto append=[&](OptixTraversableHandle handle,const m3r4c& local,uint32_t id,uint8_t mask=EverythingMask) {
            if(!handle)return;
            OptixInstance instance{};instance.traversableHandle=handle;instance.instanceId=id;instance.visibilityMask=mask;
            const auto world=composeAffine(camera,local);memcpy(instance.transform,world.data(),sizeof(instance.transform));
            if(zeno::mega::visible(world))collector.append(instance);
        };
        if(lightsWrapper.lightIasHandle) {
            append(lightsWrapper.lightIasHandle,IdentityMatrix,0,LightMatMask);maxNodeDepth=std::max(maxNodeDepth,uint8_t(3));
        }
        auto appendGroup=[&](uint64_t handle,const std::optional<OptixInstance>& folded,uint32_t id) {
            if(!handle)return;
            if(!folded){append(handle,IdentityMatrix,id);return;}
            auto instance=*folded;m3r4c local;memcpy(local.data(),instance.transform,sizeof(instance.transform));
            const auto world=composeAffine(camera,local);memcpy(instance.transform,world.data(),sizeof(instance.transform));
            collector.append(instance);
        };
        if(fusedRootGroup) {
            auto outer=IdentityMatrix;
            const auto& folded=staticRenderGroup?staticRootInstance:dynamicRootInstance;
            if(folded)memcpy(outer.data(),folded->transform,sizeof(folded->transform));
            const auto rootTransform=composeAffine(camera,outer);
            collector.reserveAdditional(fusedRootGroup->hostInstances.size());
            for(auto instance:fusedRootGroup->hostInstances) {
                m3r4c local;memcpy(local.data(),instance.transform,sizeof(instance.transform));
                const auto world=composeAffine(rootTransform,local);memcpy(instance.transform,world.data(),sizeof(instance.transform));
                collector.append(instance);
            }
        } else {
            appendGroup(staticRenderGroup,staticRootInstance,1);
            appendGroup(dynamicRenderGroup,dynamicRootInstance,2);
        }
        // Preserve direct VIEW-node graphs, but grouped resources enter through
        // their actual scene parents and must never also appear under root.
        for(const auto& [name,source]:streamedGeometry)if(source.handle && !streamedGeometryInScene(name)) {
            if(name==clusterObjectName)for(const auto& placement:clusterPlacements)append(source.handle,placement,3);
            else append(source.handle,IdentityMatrix,4);
            maxNodeDepth=std::max(maxNodeDepth,uint8_t(source.depth+1));
        }
        // Child identity/layout decides UPDATE legality; bounds versions decide
        // whether it is necessary. Stable handles still require ancestor refits.
        xinxinoptix::buildIAS(context,rootNode.hostInstances,rootNode,collector.canUpdate(),true);
        collector.commit();rootNode.frame=frameid;rootNode.valid=true;rootCamera=camera;rootBoundsDirty=false;
        clusterRootDirty=false;waterRootDirty=false;
    }

    uint8_t maxNodeDepth = 1;
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCache {};
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodeCacheStatic {};
    bool collapseDynamicSingleChildIAS{true};

    float3 cameraPos;
    SceneNodeT<CudaPinnedAllocator> rootNode;
    IASBuildWorkspace iasBuildWorkspace;

    uint64_t staticRenderGroup {};
    uint64_t dynamicRenderGroup {};
    std::optional<OptixInstance> staticRootInstance,dynamicRootInstance;
    
    std::unordered_map<std::string, Candidate> candidates {};
    bool volume_scene_bindings_dirty {};

    void preload_mesh(const std::string &key, const std::string &mtlid, 
        const std::vector<std::string> &matNames, const int *matIds,
        const float *verts, size_t numverts, const uint *tris, size_t numtris,
        std::map<std::string, std::pair<float const *, size_t>> const &vtab, const GeoChange change=GeoChange::FullChange);

    void unload_object(std::string const &key) {
        if(waterSettings.count(key))waterUnloadPending.insert(key);
        if(clusterAssetKey==key) {
            clusterUnloadPending=true;
        }
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
        rootBoundsDirty=true;
        for (auto* cache:{&nodeCache,&nodeCacheStatic})for(auto& [_,node]:*cache) {
            if (!node) continue;
            node->valid = false;
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

        finalizeSceneRegistration();

        std::map<std::string, std::set<ShaderMark>> shader_key_set;

        uniqueMatsForMesh.insert("");
        for (auto& mat : uniqueMatsForMesh) {
            auto& cached = shader_key_set[mat];
            cached.insert( ShaderMark::Mesh );
        }
        for(const auto& [_,material]:waterMaterialNames)
            shader_key_set[material].insert(ShaderMark::Mesh);

        if (sceneJson.contains(brikey)) {
            auto& bri = sceneJson[brikey];

            for (auto it = bri.begin(); it != bri.end(); ++it) {

                std::string geo_name = it.value()["Geom"].template get<std::string>();
                if(isStreamedGeometry(geo_name))continue;
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
