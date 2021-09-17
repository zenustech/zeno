#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <tetrahedra_mesh.h>

namespace {

using namespace zeno;

struct FEMObject : zeno::IObject{
    FEMObject() = default;
    FEMObject(const std::vector<FEM_Scaler>& _vertices,const std::vector<size_t> _elements){
        vertices = _vertices;
        elements = elements;
    }

    std::vector<FEM_Scaler> vertices;
    std::vector<size_t> elements;
    std::vector<size_t> cons_dofs;
    std::vector<size_t> uncons_dofs;
    std::vector<size_t> surf_dofs;

    std::vector<FEM_Scaler> vert_mass;
    std::vector<FEM_Scaler> dof_mass;
    std::vector<FEM_Scaler> elm_volume;
    std::vector<FEM_Scaler> elm_mass;

    std::vector<Mat9x12d> elm_dFdx;
    std::vector<Mat3x3d> elm_WBmT;

    std::vector<Mat3x3d> elm_anisotropic_orients;
    std::vector<Vec3d> elm_anisotropic_weights;
    std::vector<Mat3x3d> elm_activation;


    void DoPreComputation() {
        
    }
};


struct MakeFEMObjFromFile : zeno::INode{
    virtual void apply() override {
        auto NodeFile = get_input<zeno::StringObject>("NodeFile")->get();
        auto EleFile = get_input<zeno::StringObject>("EleFile")->get();    

    }
}

ZENDEFNODE(MakeFEMObjFromFile, {
    {{"readpath","NodeFile"},{"readpath", "EleFile"},{"readpath", "BCFile"}},
    {"FEMObj", "FEMGeo"},
    {},
    {"FEM"},
});

struct MakeFEMObjFromFile : zeno::INode {
    virtual void apply() override {
        auto NodeFile = get_input<zeno::StringObject>("NodeFile")->get();
        std::cout<<NodeFile<<std::endl;
        auto EleFile = get_input<zeno::StringObject>("EleFile")->get();
        std::cout<<EleFile<<std::endl;
        const char* BCFileStr = nullptr;
        // if(has_input("BCFile")){
            auto BCFile = get_input<zeno::StringObject>("BCFile")->get();
        //     std::cout<<BCFile<<std::endl;
        //     BCFileStr = BCFile.c_str();
        // }

        auto res = std::make_shared<FEMObject>();
        TetrahedraMesh::LoadTetrahedraFromFile(res->FEMData_ptr, EleFile.c_str(),NodeFile.c_str(),BCFile.c_str());

        auto resGeo =  std::make_shared<zeno::PrimitiveObject>();
        auto &pos = resGeo->add_attr<zeno::vec3f>("pos");

        for(int i=0;i<res->FEMData_ptr->GetNumVertices();++i){
            auto vert = res->FEMData_ptr->GetVertex(i);
            pos.emplace_back(zeno::vec3f(vert[0],vert[1],vert[2]));
        }

        for(int i=0;i < res->FEMData_ptr->GetNumElements();++i){
            auto tet = res->FEMData_ptr->GetElement(i);
            resGeo->tris.emplace_back(tet[0],tet[1],tet[2]);
            resGeo->tris.emplace_back(tet[1],tet[3],tet[2]);
            resGeo->tris.emplace_back(tet[0],tet[2],tet[3]);
            resGeo->tris.emplace_back(tet[0],tet[3],tet[1]);
        }

        pos.resize(res->FEMData_ptr->GetNumVertices());
        set_output("FEMObj",std::move(res));
        set_output("FEMGeo", std::move(resGeo));
    }
};

ZENDEFNODE(MakeFEMObjFromFile, {
    {{"readpath","NodeFile"},{"readpath", "EleFile"},{"readpath", "BCFile"}},
    {"FEMObj", "FEMGeo"},
    {},
    {"FEM"},
});

struct MuscleForceModel : zeno::INode {

}




}