#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>

#include "assimp/scene.h"

#include "Definition.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace {

struct ExtractFBXData : zeno::INode {

    virtual void apply() override {

        auto fbxData = get_input<FBXData>("data");

        set_output("vertices", std::move(fbxData->iVertices.clone()));
        set_output("indices", std::move(fbxData->iIndices.clone()));
        set_output("material", std::move(fbxData->iMaterial.clone()));
        set_output("boneOffset", std::move(fbxData->iBoneOffset.clone()));
    }
};
ZENDEFNODE(ExtractFBXData,
           {       /* inputs: */
               {
                   {"FBXData", "data"},
               },  /* outputs: */
               {
                   {"IVertices", "vertices"},
                   {"IIndices", "indices"},
                   {"IMaterial", "material"},
                   {"IBoneOffset", "boneOffset"}
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct ExtractMatName : zeno::INode {

    virtual void apply() override {
        auto mat = get_input<IMaterial>("material");
        auto key = get_input<zeno::StringObject>("key")->get();
        auto name = std::make_shared<zeno::StringObject>();

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
                "material", "key"
            },  /* outputs: */
            {
                "name"
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

struct ExtractMatTexList : zeno::INode {

    virtual void apply() override {
        auto mat = get_input<SMaterial>("material");

        //zeno::log_info(">>>>> Get Key {}", key);
        //zeno::log_info(">>>>> Get Mat Name {}", mat->value.at(key).matName);

        auto name = std::make_shared<zeno::StringObject>();

        auto lo = std::make_shared<zeno::ListObject>();
        auto tl = mat->getTexList();
        for(auto&p: tl){
            auto s = std::make_shared<zeno::StringObject>();
            s->value = p;
            lo->emplace_back(s);
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
                   {"IMaterial", "material"},
               },  /* outputs: */
               {
                    "texLists", "name"
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct ExtractMatDict : zeno::INode {

    virtual void apply() override {
        auto mat = get_input<IMaterial>("material");
        auto mats = std::make_shared<zeno::DictObject>();
        for(auto& m:mat->value){
            auto name = m.second.matName;
            if(mats->lut.find(name) == mats->lut.end()){
                auto sm = std::make_shared<SMaterial>(m.second);
                mats->lut[name] = sm;
            }
        }
        set_output("mats", std::move(mats));
    }
};
ZENDEFNODE(ExtractMatDict,
           {       /* inputs: */
            {
                {"IMaterial", "material"},
            },  /* outputs: */
            {
                "mats",
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

struct ExtractMatData : zeno::INode {

    virtual void apply() override {
        auto data = get_input<MatData>("data");
        auto datas = std::make_shared<zeno::ListObject>();
        auto matName = std::make_shared<zeno::StringObject>();

//        TIMER_START(MakeDatas)
        for(auto [k, v]: data->iFbxData.value){
            datas->push_back(v);
        }
        matName->set(data->sMaterial.matName);
//        TIMER_END(MakeDatas)

        // Make Zeno Objects
        auto texLists = std::make_shared<zeno::ListObject>();
        auto texMaps = std::make_shared<zeno::DictObject>();
        auto texUvs = std::make_shared<zeno::DictObject>();
        auto matValues = std::make_shared<zeno::DictObject>();

        // Get Texture List And Prop-Index And Mat Params Value
        std::vector<std::string> texList{};
        std::map<std::string, int> texMap{};
        std::map<std::string, aiUVTransform> texUv{};
        std::map<std::string, aiColor4D> matValue{};

//        TIMER_START(SetMats)
        data->sMaterial.getSimplestTexList(texList, texMap, texUv, matValue);

        // Set Data -> Zeno Object
        for(auto& path: texList){
            auto strObj = std::make_shared<zeno::StringObject>();
            strObj->value = path;
            texLists->emplace_back(strObj);
        }
        for(auto&[matPropName, index]: texMap){
            auto numeric_obj = std::make_shared<zeno::NumericObject>();
            numeric_obj->set(index);
            texMaps->lut[matPropName] = std::move(numeric_obj);
        }

        for(auto&[matPropName, uvTrans]: texUv){
            auto numeric_obj = std::make_shared<zeno::NumericObject>();
            numeric_obj->set(zeno::vec4f(uvTrans.mScaling.x, uvTrans.mScaling.y, uvTrans.mTranslation.x, uvTrans.mTranslation.y));
            texUvs->lut[matPropName] = std::move(numeric_obj);
        }

        for(auto& [matPropName, matPropValue]: matValue){
            auto numeric_obj = std::make_shared<zeno::NumericObject>();
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
                "data"
            },  /* outputs: */
            {
                {"list", "datas", ""},
                "matName",
                {"list", "texLists", ""},
                {"dict", "texMaps", ""},
                {"dict", "matValues", ""},
                {"dict", "texUvs", ""}
            },  /* params: */
            {

            },  /* category: */
            {
                "FBX",
            }
           });

struct ExtractCameraData : zeno::INode {

    virtual void apply() override {
        auto icam = get_input<ICamera>("camobject");
        auto key = get_input<zeno::StringObject>("key")->get();
        if(icam->value.find(key) == icam->value.end()){
            throw zeno::makeError("Camera Not Found");
        }

        auto cam = icam->value.at(key);

        auto pos = std::make_shared<zeno::NumericObject>();
        auto up = std::make_shared<zeno::NumericObject>();
        auto view = std::make_shared<zeno::NumericObject>();
        auto focL = std::make_shared<zeno::NumericObject>();
        auto filmW = std::make_shared<zeno::NumericObject>();
        auto filmH = std::make_shared<zeno::NumericObject>();
        auto haov = std::make_shared<zeno::NumericObject>();
        auto waov = std::make_shared<zeno::NumericObject>();
        auto hfov = std::make_shared<zeno::NumericObject>();
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
           {"string", "key", "camera1"}, "camobject"
               },  /* outputs: */
               {
                   "pos", "up", "view", "focL", "haov", "waov", "hfov", "filmW", "filmH"
               },  /* params: */
               {

               },  /* category: */
               {
                   "FBX",
               }
           });

struct ExchangeFBXData : zeno::INode {

    virtual void apply() override {
        auto animinfo = get_input<AnimInfo>("animinfo");
        auto nodetree = get_input<NodeTree>("nodetree");
        auto bonetree = get_input<BoneTree>("bonetree");

        auto paramDType = get_param<std::string>("dType");
        std::string dType;
        if(paramDType == "DATA"){
            auto data = get_input<FBXData>("d");
            data->nodeTree = nodetree;
            data->boneTree = bonetree;
            data->animInfo = animinfo;
            set_output("d", std::move(data));
        }else if(paramDType == "DATAS"){
            auto datas = get_input<zeno::DictObject>("d");
            for (auto &[k, v]: datas->lut) {
                auto vc = zeno::safe_dynamic_cast<FBXData>(v.get());
                vc->animInfo = animinfo;
                vc->boneTree = bonetree;
                vc->nodeTree = nodetree;
            }
            set_output("d", std::move(datas));
        }else if(paramDType == "MATS"){
            auto mats = get_input<zeno::DictObject>("d");
            for (auto &[k, v]: mats->lut) {
                auto vc = zeno::safe_dynamic_cast<MatData>(v.get());
                for(auto &[_k, _v]: vc->iFbxData.value){
                    _v->animInfo = animinfo;
                    _v->nodeTree = nodetree;
                    _v->boneTree = bonetree;
                }
            }
            set_output("d", std::move(mats));
        }
    }
};
ZENDEFNODE(ExchangeFBXData,
           {       /* inputs: */
            {
                "d", "animinfo", "nodetree", "bonetree",
            },  /* outputs: */
            {
                "d",
            },  /* params: */
            {
                {"enum DATA DATAS MATS", "dType", "DATA"},
            },  /* category: */
            {
                "FBX",
            }
           });

}
