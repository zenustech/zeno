#include "declares.h"

namespace zeno {

struct MakeFEMIntegrator : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto muscle = get_input<MuscleModelObject>("elasto");
        auto damp = get_input<DampingForceModel>("visco");
        auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto inttype = std::get<std::string>(get_param("integType"));

        auto res = std::make_shared<FEMIntegrator>();
        if(inttype == "BE"){
            std::cerr << "CURRENTLY ONLY QUASI_STATIC SOLVERS ARE SUPPORTED" << std::endl;
            throw std::runtime_error("CURRENTLY ONLY QUASI_STATIC SOLVERS ARE SUPPORTED");
            res->_intPtr = std::make_shared<BackEulerIntegrator>();
        }
        else if(inttype == "QS")
            res->_intPtr = std::make_shared<QuasiStaticSolver>();
        else if(inttype == "EBQS")
            res->_intPtr = std::make_shared<ExamBasedQuasiStaticSolver>();
        else if(inttype == "SDQS")
            res->_intPtr = std::make_shared<SkinDrivenQuasiStaticSolver>();

        res->_intPtr->SetGravity(Vec3d(gravity[0],gravity[1],gravity[2]));
        res->_intPtr->SetTimeStep(dt);

        res->muscle = muscle;
        res->damp = damp;

        res->_stepID = 0;

        res->PrecomputeFEMInfo(prim);
        set_output("FEMIntegrator",res);
    }
};

ZENDEFNODE(MakeFEMIntegrator,{
    {{"prim"},{"elasto"},{"visco"},{"gravity"},{"dt"}},
    {"FEMIntegrator"},
    {{"enum BE QS EBQS SDQS", "integType", "QS"}},
    {"FEM"},
});

};