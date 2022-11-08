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

    // #########################################################
    bool changed = false;
    for (auto const &[key, obj] : objs) {
        if (zins.may_emplace(key)) {
            changed = true;
            break;
        }
    }

    if(changed){
        zenoStage->RemoveStagePrims();
        convertObjects.clear();
        objectConsistent.clear();
        zenoObjects.clear();
        zenoLightObjects.clear();
    }

    // #########################################################
    bool inserted = false;
    for (auto const &[key, obj] : objs) {
        if (zins.may_emplace(key)) {
            // Legacy Light
            // TODO Legacy light to usdLux
            auto isRealTimeObject = obj->userData().get2<int>("isRealTimeObject", 0);
            if(isRealTimeObject){
                zenoLightObjects[key] = obj;
            }

            // TODO Save userData to Usd Stage, Currently it is a local conversion
            // Prim
            std::string p_path, p_type;
            ZPrimInfo primInfo;
            obj->userData().has("P_Path") ? p_path = obj->userData().get2<std::string>("P_Path") : p_path = "";
            obj->userData().has("P_Type") ? p_type = obj->userData().get2<std::string>("P_Type") : p_type = "";
            primInfo.pPath = p_path; primInfo.iObject = obj;
            //std::cout << "USD: StageManager Emplace " << key << " Type " << p_type << " Path " << p_path << std::endl;

            if(p_type == _primTokens->UsdGeomMesh.GetString()){
                zenoStage->Convert2UsdGeomMesh(primInfo);
            }
            else if(p_type == _primTokens->UsdGeomCube.GetString()){
                zenoStage->Convert2UsdGeomMesh(primInfo);
            }
            else if(p_type == _primTokens->UsdLuxDiskLight.GetString()){

            }else{
                zeno::log_warn("USD: Unsupported type {}, name {}", p_type, key);
            }

            if(! p_path.empty()){
                ZOriginalInfo oriInfo; oriInfo.oName = key; oriInfo.oUserData = obj->userData();
                nameComparison[p_path] = oriInfo;
            }

            zins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }

    // #########################################################
    if(inserted){
        // Stage C is synthesized with stage S and output to the Layer,
        // and stage F layered the Layer
        zenoStage->CompositionArcsStage();
        zenoStage->cStagePtr->Save();
        zenoStage->fStagePtr->Reload();

        // TODO Handle path conflict situations, perhaps over, priorities, etc
        zenoStage->CheckPathConflict();
        zenoStage->TraverseStageObjects(zenoStage->fStagePtr, objectConsistent);

        increase_count++;
    }

    // #########################################################
    auto cins = convertObjects.insertPass();
    bool converted = false;
    for(auto const&[k ,p]: objectConsistent){
        std::string nk = k+":"+std::to_string(increase_count);
        if(cins.may_emplace(nk)){
            // Comparison
            // If zInfo is empty that mean the object is from AnotherStage
            auto zInfo = nameComparison[k];
            nameComparison[nk] = nameComparison[k];
            nameComparison.erase(k);
            if(! zInfo.oName.empty())
                p.iObject->userData() = zInfo.oUserData;

            cins.try_emplace(nk, std::move(p.iObject));
            converted = true;
        }
    }
    if(converted){
        std::cout << "USD: Consistent Size " << objectConsistent.size() << std::endl;
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