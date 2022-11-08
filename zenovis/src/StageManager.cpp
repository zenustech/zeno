#include <zenovis/StageManager.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>

#include "zenovis/Stage.h"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace zenovis {

StageManager::StageManager(){
    zeno::log_info("USD: StageManager Constructed");
    zenoStage = std::make_shared<ZenoStage>();

};
StageManager::~StageManager(){
    zeno::log_info("USD: StageManager Destroyed");
};

bool zenovis::StageManager::load_objects(const std::map<std::string, std::shared_ptr<zeno::IObject>> &objs) {
    auto zins = zenoObjects.insertPass();
    increase_count++;

    bool inserted = false;

    for (auto const &[key, obj] : objs) {
        if (zins.may_emplace(key)) {
            zenoLightObjects.clear();
            break;
        }
    }

    // #########################################################
    for (auto const &[key, obj] : objs) {
        if (zins.may_emplace(key)) {
            // Legacy Light
            // TODO Legacy light to usdLux
            auto isRealTimeObject = obj->userData().get2<int>("isRealTimeObject", 0);
            if(isRealTimeObject){
                zenoLightObjects[key] = obj;
            }

            // Prim
            std::string p_path, p_type;
            ZPrimInfo primInfo;
            obj->userData().has("P_Path") ? p_path = obj->userData().get2<std::string>("P_Path") : p_path = "";
            obj->userData().has("P_Type") ? p_type = obj->userData().get2<std::string>("P_Type") : p_type = "";
            primInfo.pPath = p_path; primInfo.iObject = obj;
            zeno::log_info("USD: StageManager Emplace {}, P_Type {}, P_Path {}", key, p_type, p_path);

            if(p_type == _tokens->UsdGeomMesh.GetString()){
                zenoStage->Convert2UsdGeomMesh(primInfo);
            }
            else if(p_type == _tokens->UsdGeomCube.GetString()){
                zenoStage->Convert2UsdGeomMesh(primInfo);
            }
            else if(p_type == _tokens->UsdLuxDiskLight.GetString()){

            }else{
                zeno::log_info("USD: Unsupported type {}, name {}", p_type, key);
            }

            zins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }

    zenoStage->cStagePtr->Save();

    // #########################################################
    auto cins = convertObjects.insertPass();
    if(inserted){
        zeno::log_info("USD: Convert USD Object To ZenoObject");
        for (UsdPrim prim: zenoStage->cStagePtr->TraverseAll()) {
            // Visibility
            UsdAttribute attr_visibility = prim.GetAttribute(TfToken("visibility"));
            TfToken visibility;
            attr_visibility.Get(&visibility, 0.0);
            if (visibility == UsdGeomTokens->invisible)
                continue;

            // Mesh
            if(prim.GetTypeName() == "Mesh"){
                UPrimInfo primInfo{prim, std::make_shared<zeno::PrimitiveObject>()};
                std::string primPath = prim.GetPrimPath().GetAsString();
                consistent[primPath] = primInfo;
                zenoStage->Convert2ZenoPrimitive(primInfo);
            }
        }
    }

    // #########################################################
    bool converted = false;
    for(auto const&[k ,p]: consistent){
        if(cins.may_emplace(k)){
            cins.try_emplace(k, std::move(p.iObject));
            converted = true;
        }
    }

    return converted;
}

std::optional<zeno::IObject* > StageManager::get(std::string nid) {
    for (auto &[key, ptr]: this->pairs()) {
        if (key != nid) {
            continue;
        }
        return ptr;
    }

    return std::nullopt;
}

}