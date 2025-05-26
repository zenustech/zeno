#include "Scene.h"
#include "TypeCaster.h"
#include "optix_types.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <vector>
#include <zeno/utils/orthonormal.h>

void Scene::preload_mesh(std::string const &key, std::string const &mtlid,
    float const *verts, size_t numverts, uint const *tris, size_t numtris,
    std::map<std::string, std::pair<float const *, size_t>> const &vtab,
    int const *matids, std::vector<std::string> const &matNameList)
{
    MeshDat &dat = meshdats[key];
    dat.dirty = true;

    dat.triMats.assign(matids, matids + numtris);
    dat.mtlidList = matNameList;
    dat.mtlid = mtlid;
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

void Scene::updateDrawObjects(uint16_t sbt_count) {

    std::vector<std::string>    names;
    std::vector<const MeshDat*> candidates;

    for (auto &[key, dat]: meshdats) {

        if (!dat.dirty) { continue; }
        dat.dirty = false;

        candidates.push_back(&dat);
        names.push_back(key);
    }

    for (size_t i=0; i<candidates.size(); ++i) {
        auto& name = names[i];

        auto& mesh = _meshes_[name];
        auto& dat = *candidates[i];
        if (mesh == nullptr) {
            mesh = std::make_shared<MeshObject>();
        }
        mesh->dirty = true;
        mesh->resize(dat.tris.size()/3, dat.verts.size()/3);

        if (dat.mtlidList.size()>1) {

            mesh->mat_idx.resize(dat.tris.size()/3);

            std::vector<uint16_t> global_matidx(max(dat.mtlidList.size(), 1ull));

            for (size_t j=0; j<dat.mtlidList.size(); ++j) {
                const auto& matName = dat.mtlidList[j];
                const auto matKey = std::tuple { matName, ShaderMark::Mesh };
                auto it = shader_indice_table.find(matKey);
                global_matidx[j] = it != shader_indice_table.end() ? it->second : 0;
            }

            for (size_t i=0; i<dat.tris.size()/3; ++i) {
                int mtidx = max(0, dat.triMats[i]);
                mesh->mat_idx[i] = global_matidx[mtidx];
            }
        } else {

            mesh->mat_idx = {0};

            const auto matKey = std::tuple { dat.mtlid, ShaderMark::Mesh };
            auto it = shader_indice_table.find(matKey);

            if (it != shader_indice_table.end()) {
                auto idx = it->second;
                mesh->mat_idx[0] = idx;
            }
        }
        
        memcpy(mesh->indices.data(), dat.tris.data(), sizeof(uint)*dat.tris.size() );
        memcpy(mesh->vertices.data(), dat.verts.data(), sizeof(float)*dat.verts.size() );

        auto ver_count = dat.verts.size()/3;

        auto& uvAttr = dat.getAttr("uv");
        auto& clrAttr = dat.getAttr("clr");
        auto& nrmAttr = dat.getAttr("nrm");
        auto& tangAttr = dat.getAttr("atang");

        tbb::parallel_for(size_t(0), ver_count, [&](size_t i) {
            
            mesh->g_uv[i] = ( *(float2*)&(uvAttr[i * 3]) );
            mesh->g_clr[i] = toHalf( *(float3*)&(clrAttr[i * 3]) );

            mesh->g_nrm[i] = toHalf( *(float3*)&(nrmAttr[i * 3]) );
            mesh->g_tan[i] = toHalf( *(float3*)&(tangAttr[i * 3]) );
        });

        dirtyTasks[name] = [&](OptixDeviceContext& context){
            if (nullptr == mesh || mesh->vertices.empty()) return 0ull;

            mesh->dirty = false;
            mesh->buildGas(context, sbt_count);

            return mesh->node->handle;
        };

        cleanTasks[name] = [&](const std::string& k) {
            _meshes_.erase(k);
            meshdats.erase(k);
        };
    }

}

bool Scene::preloadVDB(const zeno::TextureObjectVDB& texVDB, std::string& combined_key)
{
    auto path = texVDB.path;
    auto channel = texVDB.channel;

    std::filesystem::path filePath = path;

    if ( !std::filesystem::exists(filePath) ) {
        std::cout << filePath.string() << " doesn't exist" << std::endl;
        return false;
    }

    auto fileTime = std::filesystem::last_write_time(filePath);

    auto isNumber = [] (const std::string& s)
    {
        for (char const &ch : s) {
        if (std::isdigit(ch) == 0)
            return false;
        }
        return true;
    };

    if ( isNumber(channel) ) {
        auto channel_index = (uint)std::stoi(channel);
        channel = fetchGridName(path, channel_index);
    } else {
        checkGridName(path, channel);
    }

    const auto vdb_key = path + "{" + channel + "}";
    combined_key = vdb_key;

    zeno::log_debug("loading VDB :{}", path);

    if (_vdb_grids_cached.count(vdb_key)) {

        auto& cached = _vdb_grids_cached[vdb_key];

        if (fileTime == cached->file_time && texVDB.eleType == cached->type) {
            return true;
        } else {
            cleanupVolume(*cached);
        }
    }

    auto volume_ptr = std::make_shared<VolumeWrapper>();
    volume_ptr->file_time = fileTime;
    volume_ptr->selected = {channel};
    volume_ptr->type = texVDB.eleType;
    volume_ptr->dirty = true;

    auto succ = loadVolume(*volume_ptr, path); 

    if (!succ) {return false;}

    _vdb_grids_cached[vdb_key] = volume_ptr;
    return true;
}

void Scene::preloadCurveGroup(std::vector<float3>& points, std::vector<float>& widths, std::vector<float3>& normals, std::vector<uint>& strands, zeno::CurveType curveType, const std::string& key) {

    auto& cg = curveGroupCache[key];
    if (nullptr == cg) {
        cg = std::make_shared<CurveGroup>();
    }
    cg->curveType = curveType;

    cg->points = std::move(points);
    cg->widths = std::move(widths);
    cg->normals = std::move(normals);
    cg->strands = std::move(strands);

    auto mark = (uint)curveType + 3;
    updateGeoType( key, ShaderMark(mark) );

    dirtyTasks[key] = [&, key](OptixDeviceContext& context) {
        cg->dirty = false;

        auto state = std::make_shared<CurveGroupWrapper>();
        state->curveGroup = cg;
        state->curveType = cg->curveType;

        state->makeCurveGroupGAS(context);
        curveGroupStateCache[key] = state;

        return state->node->handle;
    };

    cleanTasks[key] = [&](const std::string& k) {
        curveGroupCache.erase(k);
        curveGroupStateCache.erase(k);
    };
}

void Scene::preloadHair(const std::string& name, const std::string& filePath, uint mode, glm::mat4 transform) {

    auto lwt = std::filesystem::last_write_time(filePath);
    bool neo = false;

    auto hair = [&]() -> std::shared_ptr<Hair> {

        if (hairCache.count(filePath) == 0 || lwt != hairCache[filePath]->time()) 
        {
            neo = true;
            auto tmp = std::make_shared<Hair>( filePath );
            tmp->prepareWidths();
            hairCache[filePath] = tmp;
            return tmp;
        }
        return hairCache[filePath];
    } ();

    //auto key = std::tuple {filePath, mode};
    if (hairStateCache.count( name ) == 0 || neo) {

        auto& tmp = hairStateCache[name];
        if (nullptr == tmp) {
            tmp = std::make_shared<CurveGroupWrapper>();
        }
        tmp->curveType = (zeno::CurveType)mode;
        tmp->pHair = hair;

        dirtyTasks[name] = [&](OptixDeviceContext& context) {
            tmp->dirty = false;
            tmp->makeHairGAS(context);
            return tmp->node->handle;
        };

        cleanTasks[name] = [&](const std::string& k) {
            hairStateCache.erase(k);
        };
    } 
        
    geoMatrixMap[name] = glm::transpose(transform);
    auto mark = (uint)mode + 3;
    updateGeoType( name, ShaderMark(mark) );
}

void Scene::preload_sphere(const std::string &key, const glm::mat4& transform) 
{
    auto& dsphere = _spheres_[key];
    if (nullptr == dsphere) {
        dsphere = std::make_shared<SphereTransformed>();
    }
    auto trans = glm::transpose(transform);

    geoMatrixMap[key] = trans;
    updateGeoType(key, ShaderMark::Sphere);

    dirtyTasks[key] = [&](OptixDeviceContext& context){
       
        if (nullptr == uniform_sphere_gas) {
            uniform_sphere_gas = std::make_shared<SceneNode>();
            buildUnitSphereGAS(context, uniform_sphere_gas->handle, uniform_sphere_gas->buffer);
        }
        dsphere->dirty = false;
        dsphere->node = uniform_sphere_gas;

        return uniform_sphere_gas->handle;
    };

    cleanTasks[key] = [&](const std::string& k) {
        _spheres_.erase(k);
    };
}

void Scene::preload_sphere_group(const std::string& key, std::vector<zeno::vec3f>& centerV, std::vector<float>& radiusV, std::vector<zeno::vec3f>& colorV) {
        
    auto& group = _sphere_groups_[key];
    if (nullptr == group) {
        group = std::make_shared<SphereGroup>();
        group->node = std::make_shared<SceneNode>();
    }
    group->centerV = std::move(centerV);
    group->radiusV = std::move(radiusV);
    group->colorV = std::move(colorV);

    updateGeoType(key, ShaderMark::Sphere);

    dirtyTasks[key] = [&](OptixDeviceContext& context){
        group->dirty = false;
        buildSphereGroupGAS(context, *group);
        return group->node->handle;
    };

    cleanTasks[key] = [&](const std::string& k) {
        _sphere_groups_.erase(k);
    };
}

bool Scene::preloadVolumeBox(const std::string& key, std::string& matid, uint8_t bounds, glm::mat4& transform) {

    auto& vbox = _vboxs_[key];
    if (nullptr == vbox) {
        vbox = std::make_shared<VolumeWrapper>(); 
    }
    auto trans = glm::transpose(transform);

    vbox->dirty = true;
    vbox->bounds = bounds;
    vbox->transform = trans;
    updateGeoType(key, ShaderMark::Volume);
    geoMatrixMap[key] = trans;

    dirtyTasks[key] = [&](OptixDeviceContext& context) {
        
        vbox->dirty = false;
        buildVolumeAccel(*vbox, context);
        return vbox->node->handle;
    };

    cleanTasks[key] = [&](const std::string& k) {
        _vboxs_.erase(k);
    };

    return true;
}