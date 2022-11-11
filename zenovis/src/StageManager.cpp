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
    stateInfo = new HandleStateInfo;
    zenoStage = std::make_shared<ZenoStage>();
    zenoStage->stateInfo = stateInfo;
    zenoStage->init();
};
StageManager::~StageManager(){
    zeno::log_info("USD: StageManager Destroyed");
    delete stateInfo;
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
        increase_count++;
        std::cout << "USD: Objects Changed Times " << increase_count << "\n";
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

            // TODO keep userData to Usd Stage, Currently it is a local conversion
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
        TIMER_START(CompositionArcsStage)
        zenoStage->CompositionArcsStage();
        zenoStage->fStagePtr->Reload();

        // TODO Handle path conflict situations, perhaps over, priorities, etc
        zenoStage->CheckPathConflict();
        TIMER_END(CompositionArcsStage)
    }
    TIMER_START(TraverseStage)
    objectConsistent.clear();
    zenoStage->TraverseStageObjects(zenoStage->fStagePtr, objectConsistent);
    TIMER_END(TraverseStage)

    {
        //auto prim = std::make_shared<zeno::PrimitiveObject>();
        //prim->verts.emplace_back(zeno::vec3f(debug_count/10.0f,0,0));
        //prim->verts.emplace_back(zeno::vec3f(0,0,0));
        //prim->verts.emplace_back(zeno::vec3f(0,1,0));
        //prim->tris.emplace_back(zeno::vec3f(0,1,2));
        //UPrimInfo info;
        //info.iObject = prim;
        //objectConsistent["___debug"+std::to_string(debug_count)] = info;
    }

    //for(auto const&[k ,p]: objectConsistent){
    //    auto obj = p.iObject;
    //    if (obj->has_attr("pos")) {
    //        auto &pos = obj->attr<zeno::vec3f>("pos");
    //        for (auto &po : pos) {
    //            po = po+zeno::vec3f(debug_count/50.0f, 0,0);
    //            std::cout << "po " << po[0] << ", " << po[1] << ", " << po[2] << "\n";
    //        }
    //    }
    //}

    // #########################################################
    // FIXME Run the same scene several times and the polygon will get a few errors
    auto cins = convertObjects.insertPass();
    bool converted = false;
    for(auto const&[k ,p]: objectConsistent){
        std::string nk = k+":"+std::to_string(increase_count);
        if(cins.may_emplace(nk)){
            // Comparison
            // If zInfo is empty that mean the object is from AnotherStage
            std::cout << "nk " << nk << "\n";
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

    debug_count++;

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