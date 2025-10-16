#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>

#include "assimp/scene.h"
#include <any>
#include "Definition.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace {

struct ExtractFBXData : zeno::INode {

    virtual void apply() override {

        auto fbxData = zeno::safe_dynamic_cast<FBXData>(get_input("data"));

        set_output("vertices", std::move(fbxData->iVertices.clone()));
        set_output("indices", std::move(fbxData->iIndices.clone()));
        set_output("material", std::move(fbxData->iMaterial.clone()));
        set_output("boneOffset", std::move(fbxData->iBoneOffset.clone()));
    }
};
ZENDEFNODE(ExtractFBXData,
           {       /* inputs: */
               {
                   {gParamType_Unknown, "data"},
               },  /* outputs: */
               {
                   {gParamType_Unknown, "vertices"},
                   {gParamType_Unknown, "indices"},
                   {gParamType_Unknown, "material"},
                   {gParamType_Unknown, "boneOffset"}
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct ExtractMatName : zeno::INode {

    virtual void apply() override {
        auto mat = zeno::safe_dynamic_cast<IMaterial>(get_input("material"));
        auto key = zsString2Std(get_input2_string("key"));
        auto name = std::make_unique<zeno::StringObject>();

        if(mat->value.find(key) == mat->value.end()){
            zeno::log_error("FBX: ExtractMat {} Not Found", key);
            for(auto& m:mat->value){
                zeno::log_info("FBX: ----- {} {}", m.first, m.second.matName);
            }

        }

        auto mat_name = mat->value.at(key).matName;
        //zeno::log_error("FBX: Extract Mat Name {} {}", key, mat_name);
        name->set(mat_name);

        set_output("name", std::move(name));
    }
};
ZENDEFNODE(ExtractMatName,
           {       /* inputs: */
            {
                {gParamType_Unknown, "material"}, {gParamType_String, "key"}
            },  /* outputs: */
            {
                {gParamType_String,"name"}
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

struct ExtractMatTexList : zeno::INode {

    virtual void apply() override {
        auto mat = zeno::safe_dynamic_cast<SMaterial>(get_input("material"));

        //zeno::log_info(">>>>> Get Key {}", key);
        //zeno::log_info(">>>>> Get Mat Name {}", mat->value.at(key).matName);

        auto name = std::make_unique<zeno::StringObject>();

        auto lo = std::make_unique<zeno::ListObject>();
        auto tl = mat->getTexList();
        for(auto&p: tl){
            auto s = std::make_unique<zeno::StringObject>();
            s->value = p;
            lo->push_back(std::move(s));
        }

        //for(auto&l: lo->arr){
        //    zeno::log_info("Tex: {}", std::any_cast<std::string>(l));
        //}
        //zeno::log_info(">>>>> Get TexLen {}", lo->size());

        name->set(mat->matName);

        set_output("name", std::move(name));
        set_output("texLists", std::move(lo));
    }
};
ZENDEFNODE(ExtractMatTexList,
           {       /* inputs: */
               {
                   {gParamType_Unknown, "material"},
               },  /* outputs: */
               {
                    {gParamType_List, "texLists"}, {gParamType_String,"name"}
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct ExtractMatDict : zeno::INode {

    virtual void apply() override {
        auto mat = zeno::safe_dynamic_cast<IMaterial>(get_input("material"));
        auto mats = zeno::create_DictObject();
        for(auto& m:mat->value){
            auto name = m.second.matName;
            if(mats->lut.find(name) == mats->lut.end()){
                auto sm = std::make_unique<SMaterial>(m.second);
                mats->lut[name] = std::move(sm);
            }
        }
        set_output("mats", std::move(mats));
    }
};
ZENDEFNODE(ExtractMatDict,
           {       /* inputs: */
            {
                {gParamType_Unknown, "material"},
            },  /* outputs: */
            {
                {gParamType_Dict, "mats"},
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

struct ExtractMatData : zeno::INode {

    virtual void apply() override {
        auto data = zeno::safe_dynamic_cast<MatData>(get_input("data"));
        auto datas = zeno::create_ListObject();
        auto matName = std::make_unique<zeno::StringObject>();

//        TIMER_START(MakeDatas)
        for(auto [k, v]: data->iFbxData.value){
            datas->push_back(std::move(v->clone()));
        }
        matName->set(data->sMaterial.matName);
//        TIMER_END(MakeDatas)

        // Make Zeno Objects
        auto texLists = std::make_unique<zeno::ListObject>();
        auto texMaps = std::make_unique<zeno::DictObject>();
        auto texUvs = std::make_unique<zeno::DictObject>();
        auto matValues = std::make_unique<zeno::DictObject>();

        // Get Texture List And Prop-Index And Mat Params Value
        std::vector<std::string> texList{};
        std::map<std::string, int> texMap{};
        std::map<std::string, aiUVTransform> texUv{};
        std::map<std::string, aiColor4D> matValue{};

//        TIMER_START(SetMats)
        data->sMaterial.getSimplestTexList(texList, texMap, texUv, matValue);

        // Set Data -> Zeno Object
        for(auto& path: texList){
            auto strObj = std::make_unique<zeno::StringObject>();
            strObj->value = path;
            texLists->push_back(std::move(strObj));
        }
        for(auto&[matPropName, index]: texMap){
            auto numeric_obj = std::make_unique<zeno::NumericObject>();
            numeric_obj->set(index);
            texMaps->lut[matPropName] = std::move(numeric_obj);
        }

        for(auto&[matPropName, uvTrans]: texUv){
            auto numeric_obj = std::make_unique<zeno::NumericObject>();
            numeric_obj->set(zeno::vec4f(uvTrans.mScaling.x, uvTrans.mScaling.y, uvTrans.mTranslation.x, uvTrans.mTranslation.y));
            texUvs->lut[matPropName] = std::move(numeric_obj);
        }

        for(auto& [matPropName, matPropValue]: matValue){
            auto numeric_obj = std::make_unique<zeno::NumericObject>();
            numeric_obj->set(zeno::vec3f(matPropValue.r, matPropValue.g, matPropValue.b));
            matValues->lut[matPropName] = std::move(numeric_obj);
        }
//        TIMER_END(SetMats)

        set_output("datas", std::move(datas));
        set_output("matName", std::move(matName));
        set_output("texLists", std::move(texLists));
        set_output("texMaps", std::move(texMaps));
        set_output("texUvs", std::move(texUvs));
        set_output("matValues", std::move(matValues));
    }
};
ZENDEFNODE(ExtractMatData,
           {       /* inputs: */
            {
                {gParamType_Unknown, "data"}
            },  /* outputs: */
            {
                {gParamType_List, "datas", ""},
                {gParamType_String,"matName"},
                {gParamType_List, "texLists", ""},
                {gParamType_Dict, "texMaps", ""},
                {gParamType_Dict, "matValues", ""},
                {gParamType_Dict, "texUvs", ""}
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

struct ExtractCameraData : zeno::INode {

    virtual void apply() override {
        auto icam = zeno::safe_dynamic_cast<ICamera>(get_input("camobject"));
        auto key = zsString2Std(get_input2_string("key"));
        if(icam->value.find(key) == icam->value.end()){
            throw zeno::makeError("Camera Not Found");
        }

        auto cam = icam->value.at(key);

        auto pos = std::make_unique<zeno::NumericObject>();
        auto up = std::make_unique<zeno::NumericObject>();
        auto view = std::make_unique<zeno::NumericObject>();
        auto focL = std::make_unique<zeno::NumericObject>();
        auto filmW = std::make_unique<zeno::NumericObject>();
        auto filmH = std::make_unique<zeno::NumericObject>();
        auto haov = std::make_unique<zeno::NumericObject>();
        auto waov = std::make_unique<zeno::NumericObject>();
        auto hfov = std::make_unique<zeno::NumericObject>();
        pos->set<zeno::vec3f>(cam.pos);
        up->set<zeno::vec3f>(cam.up);
        view->set<zeno::vec3f>(cam.view);
        focL->set<float>(cam.focL);
        filmW->set<float>(cam.filmW);
        filmH->set<float>(cam.filmH);
        hfov->set<float>(cam.hFov * (180.0f / M_PI));
        // Angle of view (in degrees) = 2 ArcTan( sensor width / (2 X focal length)) * (180/π)
        haov->set<float>(2.0f * std::atan(cam.filmH / (2.0f * cam.focL) ) * (180.0f / M_PI));
        waov->set<float>(2.0f * std::atan(cam.filmW / (2.0f * cam.focL) ) * (180.0f / M_PI));

        auto _pos = pos->get<zeno::vec3f>();
        auto _up = up->get<zeno::vec3f>();
        auto _view = view->get<zeno::vec3f>();
        //zeno::log_info(">>>>> P {: f} {: f} {: f}", _pos[0], _pos[1], _pos[2]);
        //zeno::log_info(">>>>> U {: f} {: f} {: f}", _up[0], _up[1], _up[2]);
        //zeno::log_info(">>>>> V {: f} {: f} {: f}", _view[0], _view[1], _view[2]);
        //zeno::log_info(">>>>> FL {: f} FW {: f} FH {: f} AOV {: f} {: f} FOV {: f} {: f}",
        //               focL->get<float>(), filmW->get<float>(), filmH->get<float>(),
        //               haov->get<float>(), waov->get<float>(), hfov->get<float>(), cam.hFov);
        //zeno::log_info("-------------------------");

        set_output("pos", std::move(pos));
        set_output("up", std::move(up));
        set_output("view", std::move(view));
        set_output("focL", std::move(focL));
        set_output("haov", std::move(haov));
        set_output("waov", std::move(waov));
        set_output("hfov", std::move(hfov));
        set_output("filmW", std::move(filmW));
        set_output("filmH", std::move(filmH));
    }
};
ZENDEFNODE(ExtractCameraData,
           {       /* inputs: */
               {
                    {gParamType_String, "key", "camera1"}, {gParamType_Unknown, "camobject"}
               },  /* outputs: */
               {
                   {gParamType_Vec3f,"pos"}, 
                   {gParamType_Vec3f,"up"}, 
                   {gParamType_Vec3f,"view"}, 
                   {gParamType_Float,"focL"}, 
                   {gParamType_Float,"haov"}, 
                   {gParamType_Float,"waov"}, 
                   {gParamType_Float,"hfov"},
                   {gParamType_Float, "filmW"}, 
                   {gParamType_Float,"filmH"}
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct ExchangeFBXData : zeno::INode {

    virtual void apply() override {
        auto animinfo = zeno::safe_uniqueptr_cast<AnimInfo>(clone_input("animinfo"));
        auto nodetree = zeno::safe_uniqueptr_cast<NodeTree>(clone_input("nodetree"));
        auto bonetree = zeno::safe_uniqueptr_cast<BoneTree>(clone_input("bonetree"));

        auto paramDType = get_param_string("dType");
        std::string dType;
        if(paramDType == "DATA"){
            auto data = zeno::safe_uniqueptr_cast<FBXData>(clone_input("d"));
            data->nodeTree = std::move(nodetree);
            data->boneTree = std::move(bonetree);
            data->animInfo = std::move(animinfo);
            set_output("d", std::move(data));
        }
        else if(paramDType == "DATAS"){
            auto datas = get_input_DictObject("d");
            for (auto &[k, v]: datas->lut) {
                auto vc = zeno::safe_uniqueptr_cast<FBXData>(v->clone());
                vc->animInfo = std::move(animinfo);
                vc->boneTree = std::move(bonetree);
                vc->nodeTree = std::move(nodetree);
            }
            set_output("d", datas->clone());
        }
        else if(paramDType == "MATS"){
            auto mats = get_input_DictObject("d");
            for (auto &[k, v]: mats->lut) {
                auto vc = zeno::safe_dynamic_cast<MatData>(v.get());
                for(auto &[_k, _v]: vc->iFbxData.value){
                    _v->animInfo = std::move(animinfo);
                    _v->nodeTree = std::move(nodetree);
                    _v->boneTree = std::move(bonetree);
                }
            }
            set_output("d", mats->clone());
        }
    }
};
ZENDEFNODE(ExchangeFBXData,
           {       /* inputs: */
            {
                {gParamType_Unknown, "d"}, {gParamType_Unknown, "animinfo"}, {gParamType_Unknown, "nodetree"}, {gParamType_Unknown, "bonetree"},
            },  /* outputs: */
            {
                {gParamType_Unknown, "d"},
            },  /* params: */
            {
                {"enum DATA DATAS MATS", "dType", "DATA"},
            },  /* category: */
            {
                "FBX",
            }
           });

}
