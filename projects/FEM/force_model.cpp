#include "declares.h"

namespace zeno {
    
struct MakeElasticForceModel : zeno::INode {
    virtual void apply() override {
        auto model_type = get_param<std::string>("ForceModel"));
        auto aniso_strength = get_param<float>("aniso_strength");
        auto res = std::make_shared<MuscleModelObject>();
        if(model_type == "Fiberic"){
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableAnisotropicMuscle(aniso_strength));
        }
        else if(model_type == "HyperElastic")
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableIsotropicMuscle());
        else if(model_type == "BSplineModel"){
            FEM_Scaler default_E = 1e7;
            FEM_Scaler default_nu = 0.499;
            res->_forceModel = std::shared_ptr<BaseForceModel>(new BSplineIsotropicMuscle(default_E,default_nu));
        }else if(model_type == "Stvk"){
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableStvk());
        }
        else{
            std::cerr << "UNKNOWN MODEL_TYPE" << std::endl;
            throw std::runtime_error("UNKNOWN MODEL_TYPE");
        }
        set_output("ElasticModel",res);
    }
};

ZENDEFNODE(MakeElasticForceModel, {
    {},
    {"ElasticModel"},
    {{"enum HyperElastic Fiberic BSplineModel Stvk", "ForceModel", "HyperElastic"},{"float","aniso_strength","20"}},
    {"FEM"},
});


struct MakeDampingForceModel : zeno::INode {
    virtual void apply() override {
        auto res = std::make_shared<DampingForceModel>();
        res->_dampForce = std::make_shared<DiricletDampingModel>();
        set_output("DampForceModel",res);
    }    
};

ZENDEFNODE(MakeDampingForceModel, {
    {},
    {"DampForceModel"},
    {},
    {"FEM"},
});

};