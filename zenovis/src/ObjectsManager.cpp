#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PreparedFrame.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <set>

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs, std::string& runtype, bool packMatrices) {
    const std::string batchKey="__zeno_optix_prepared_matrices";
    const auto previous=objects.m_curr.find(batchKey);
    auto oldBatch=previous==objects.m_curr.end()?nullptr:std::dynamic_pointer_cast<zeno::PreparedMatrixBatch>(previous->second);
    auto batch=std::make_shared<zeno::PreparedMatrixBatch>();
    auto ins=objects.insertPass();
    bool inserted=false;
    std::map<std::string,std::shared_ptr<zeno::IObject>> nextLights;
    for(const auto& [key,obj]:objs) {
        if(!obj)continue;
        if(packMatrices && key.find(":static:")==std::string::npos) {
            auto matrix=std::dynamic_pointer_cast<zeno::PreparedMatrix>(obj);
            if(matrix && !matrix->userData().has("ShaderUniforms") && !matrix->userData().get2<int>("isRealTimeObject",0)) {
                batch->matrices.emplace_back(key,std::move(matrix));continue;
            }
        }
        if(auto* primitive=dynamic_cast<zeno::PrimitiveObject*>(obj.get())) {
            if(primitive->userData().get2<int>("isRealTimeObject",0))nextLights.emplace(key,obj);
        }
        if(!ins.may_emplace(key))continue;
        const auto& ud=obj->userData();
        if(ud.get2<std::string>("ResourceType","")=="Mesh" && ud.has<std::string>("ObjectName"))
            cached_mesh[ud.get2<std::string>("ObjectName")]=obj;
        ins.try_emplace(key,obj);inserted=true;
    }
    if(!batch->matrices.empty()) {
        if(oldBatch && batch->matrices==oldBatch->matrices)batch=oldBatch;
        if(ins.may_emplace(batchKey))ins.try_emplace(batchKey,batch);
        else objects.m_next.m_curr[batchKey]=batch;
        inserted|=batch!=oldBatch;
    } else inserted|=oldBatch!=nullptr;
    if(runtype=="LoadAsset" || runtype=="RunAll" || runtype=="RunLightCamera")lightObjects=std::move(nextLights);
    else for(auto& [key,obj]:nextLights)lightObjects[key]=std::move(obj);
    return inserted;
}

void ObjectsManager::clear_objects() {
    objects.clear();
    lightObjects.clear();
}

std::optional<zeno::IObject* > ObjectsManager::get(std::string nid) {
    for (auto &[key, ptr]: this->pairs()) {
        if(auto* batch=dynamic_cast<zeno::PreparedMatrixBatch*>(ptr)) {
            for(const auto& [sourceKey,matrix]:batch->matrices)
                if(sourceKey==nid || matrix->matrixName==nid)return matrix.get();
            continue;
        }
        if (key != nid && ptr->userData().get2<std::string>("ObjectName", key) != nid) {
            continue;
        }
        return ptr;
    }
    if (cached_mesh.count(nid)) {
        return cached_mesh[nid].get();
    }

    return std::nullopt;
}

}
