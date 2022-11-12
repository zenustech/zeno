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
    //TIMER_START(TraverseStage)
    objectConsistent.clear();
    zenoStage->TraverseStageObjects(zenoStage->fStagePtr, objectConsistent);
    //TIMER_END(TraverseStage)

    bool synced = false;
    for(const auto& primPath: zenoStage->syncedObject) {
        // FIXME Error calculating results in child thread
        std::cout << "USD: Object Sync " << primPath << "\n";
        auto newPrimPath = primPath+":"+std::to_string(sync_count);
        auto& objTrans = zenoStage->objectsTransform[SdfPath(primPath)];
        UPrimInfo info; info.iObject = std::move(objectConsistent[primPath].iObject);
        auto& obj = info.iObject;
        objectConsistent[newPrimPath] = info;
        objectConsistent.erase(primPath);

        auto translate = objTrans.zTrans;
        auto rotate = objTrans.zRotate;
        auto scale = objTrans.zScale;
        auto pivot = objTrans.zPivot;
        auto lastTrans = objTrans.zLastTrans;
        auto lastRotate = objTrans.zLastRotate;
        auto lastScale = objTrans.zLastScale;

        std::cout << "USD: Trans "<<translate[0]<<","<<translate[1]<<","<<translate[2]<<"\n";
        std::cout << "USD: Rotate "<<rotate[0]<<","<<rotate[1]<<","<<rotate[2]<<"\n";
        std::cout << "USD: Scale "<<scale[0]<<","<<scale[1]<<","<<scale[2]<<"\n";
        std::cout << "USD: Pivot "<<pivot[0]<<","<<pivot[1]<<","<<pivot[2]<<"\n";

        auto transPivot = glm::translate(pivot);
        auto invTransPivot = glm::inverse(transPivot);

        // Invert last transform
        auto pre_translate_matrix = glm::translate(lastTrans);
        auto pre_scale_matrix = glm::scale(scale * lastScale);
        auto pre_transform_matrix = pre_translate_matrix * pre_scale_matrix;
        auto inv_pre_transform = glm::inverse(pre_transform_matrix);

        // Transform
        auto translate_matrix = glm::translate(translate);
        auto scale_matrix = glm::scale(scale * objTrans.zScale);
        auto transform_matrix = transPivot * translate_matrix * scale_matrix * inv_pre_transform * invTransPivot;

        auto m = transform_matrix;
        std::cout << "-----Object\n";
        std::cout <<m[0][0]<<","<<m[0][1]<<","<<m[0][2]<<","<<m[0][3] << "\n";
        std::cout <<m[1][0]<<","<<m[1][1]<<","<<m[1][2]<<","<<m[1][3] << "\n";
        std::cout <<m[2][0]<<","<<m[2][1]<<","<<m[2][2]<<","<<m[2][3] << "\n";
        std::cout <<m[3][0]<<","<<m[3][1]<<","<<m[3][2]<<","<<m[3][3] << "\n";
        std::cout << "-----\n";

        if (obj->has_attr("pos")) {
            auto &pos = obj->attr<zeno::vec3f>("pos");
            for (auto &po : pos) {
                auto p = zeno::vec_to_other<glm::vec3>(po);
                auto t = transform_matrix * glm::vec4(p, 1.0f);
                auto pt = glm::vec3(t) / t.w;
                po = zeno::other_to_vec<3>(pt);
            }
        }
        if (obj->has_attr("nrm")) {
            auto &nrm = obj->attr<zeno::vec3f>("nrm");
            for (auto &vec : nrm) {
                auto n = zeno::vec_to_other<glm::vec3>(vec);
                glm::mat3 norm_matrix(transform_matrix);
                norm_matrix = glm::transpose(glm::inverse(norm_matrix));
                auto t = glm::normalize(norm_matrix * n);
                vec = zeno::other_to_vec<3>(t);
            }
        }

        objTrans.zLastTrans = translate;
        objTrans.zLastRotate = rotate;
        objTrans.zLastScale = scale;

        synced = true;
    }
    if(synced){
        zenoStage->syncedObject.clear();
        sync_count++;
    }

    // #########################################################
    // FIXME Run the same scene several times and the polygon will get a few errors
    auto cins = convertObjects.insertPass();
    bool converted = false;
    for(auto const&[k ,p]: objectConsistent){
        std::string nk = k+":"+std::to_string(increase_count);
        if(cins.may_emplace(nk)){
            std::cout << "Emplace " << nk << " , " << k << "\n";
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