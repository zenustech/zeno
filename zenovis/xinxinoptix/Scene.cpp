#include "Scene.h"
#include "optix_types.h"
#include "zeno/utils/vec.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <vector>
#include <zeno/utils/orthonormal.h>

inline std::shared_ptr<MeshObject> StaticMeshes {};

void Scene::updateStaticDrawObjects() {
    
    size_t tri_num = 0;
    size_t ver_num = 0;

    std::vector<const MeshDat*> candidates;
    for (auto const &[key, dat]: drawdats) {
        if(key.find(":static:")!=key.npos) {
            candidates.push_back(&dat);

            tri_num += dat.tris.size()/3;
            ver_num += dat.verts.size()/3;
        }
    }

    if ( tri_num > 0 ) {
        StaticMeshes = std::make_shared<MeshObject>();
        StaticMeshes->resize(tri_num, ver_num);
    } else {
        StaticMeshes = nullptr;
        return;
    }

    size_t tri_offset = 0;
    size_t ver_offset = 0;

    for (const auto* dat_ptr : candidates) {

        auto& dat = *dat_ptr;

        std::vector<uint16_t> global_matidx(max(dat.mtlidList.size(), 1ull));

        for (size_t j=0; j<dat.mtlidList.size(); ++j) {
            auto matName = dat.mtlidList[j];
            auto it = _mesh_materials.find(matName);
            global_matidx[j] = it != _mesh_materials.end() ? it->second : 0;
        }

        tbb::parallel_for(size_t(0), dat.tris.size()/3, [&](size_t i) {
            int mtidx = dat.triMats[i];
            StaticMeshes->mat_idx[tri_offset+i] = global_matidx[mtidx];
            StaticMeshes->indices[tri_offset+i] = (*(uint3*)&dat.tris[i*3]) + make_uint3(ver_offset);
        });

        memcpy(StaticMeshes->vertices.data()+ver_offset, dat.verts.data(), sizeof(float)*dat.verts.size() );

        auto& uvAttr = dat.getAttr("uv");
        auto& clrAttr = dat.getAttr("clr");
        auto& nrmAttr = dat.getAttr("nrm");
        auto& tangAttr = dat.getAttr("atang");

        tbb::parallel_for(size_t(0), dat.verts.size()/3, [&](size_t i) {

            StaticMeshes->g_uv[ver_offset + i] = ( *(float2*)&uvAttr[i * 3] );
            StaticMeshes->g_clr[ver_offset + i] = toHalf( *(float3*)&clrAttr[i * 3] );

            StaticMeshes->g_nrm[ver_offset + i] = toHalf( *(float3*)&nrmAttr[i * 3] );
            StaticMeshes->g_tan[ver_offset + i] = toHalf( *(float3*)&tangAttr[i * 3] );
        });

        tri_offset += dat.tris.size() / 3;
        ver_offset += dat.verts.size() / 3;
    }
}

void Scene::updateDrawObjects() {

    std::vector<std::string>    names;
    std::vector<const MeshDat*> candidates;

    for (auto &[key, dat]: drawdats) {

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
                auto matName = dat.mtlidList[j];
                auto it = _mesh_materials.find(matName);
                global_matidx[j] = it != _mesh_materials.end() ? it->second : 0;
            }

            for (size_t i=0; i<dat.tris.size()/3; ++i) {
                int mtidx = max(0, dat.triMats[i]);
                mesh->mat_idx[i] = global_matidx[mtidx];
            }
        } else {

            auto matName = dat.mtlid;
            auto it = _mesh_materials.find(matName);
            uint16_t index = (it != _mesh_materials.end()) ? it->second : 0u;
            mesh->mat_idx = { index };

            if (_mesh_materials.count(dat.mtlid)>0) {
                auto idx = _mesh_materials.at(dat.mtlid);
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