#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <tetrahedra_mesh.h>
#include <fstream>
#include <algorithm>

namespace {

using namespace zeno;

struct FEMMesh : zeno::IObject{
    FEMMesh() = default;
    FEMMesh(const VecXd& _vertices,const VecXi& _elements){
        vertices = _vertices;
        elements = elements;
        extract_surf_vertices();
        compute_elm_volume();
    }

    VecXd vertices;
    VecXi elements;
    std::vector<int> surf_vertices;
    std::vector<FEM_Scaler> elm_volume;

    void compute_elm_volume() {
        size_t nm_elements = elements.size() / 4;
        elm_volume.resize(nm_elements);
    }

    typedef std::array<int,3> Triangle;

    struct TriangleCompare{
        bool operator()(const Triangle& _T1,const Triangle& _T2) const{
            auto T1 = _T1;
            auto T2 = _T2;
            std::sort(T1.begin(),T1.end(),std::greater<int>());
            std::sort(T2.begin(),T2.end(),std::greater<int>());
            if(T1[0] - T2[0] < 0)
                return true;
            if(T1[0] - T2[0] == 0 && T1[1] - T2[1] < 0)
                return true;
            if(T1[0] - T2[0] == 0 && T1[1] - T2[1] && T1[2] - T2[2] < 0)
                return true;
            return false;  
        }
    };

    void extract_surf_vertices() {
        std::set<Triangle,TriangleCompare> surf_triangles_set;

        std::vector<Triangle> triangles;
        triangles.clear();

        size_t nm_elements = elements.size() / 4;

        for(size_t i = 0;i < nm_elements;++i){
            Vec4i tet = Vec4i(elements[4*i],elements[4*i + 1],elements[4*i + 2],elements[4*i + 3]);
            surf_triangles_set.insert({tet[0],tet[1],tet[2]});
            surf_triangles_set.insert({tet[1],tet[3],tet[2]});
            surf_triangles_set.insert({tet[0],tet[2],tet[3]});
            surf_triangles_set.insert({tet[0],tet[3],tet[1]});
        }
        std::vector<Triangle> surf_triangles_vec;
        surf_triangles_vec.assign(surf_triangles_set.begin(),surf_triangles_set.end());
        surf_vertices.resize(surf_triangles_vec.size() * 3);
        for(size_t i = 0;i < surf_triangles_vec.size();++i){
            auto tri = surf_triangles_vec[i];
            surf_vertices[i*3 + 0] = tri[0];
            surf_vertices[i*3 + 1] = tri[1];
            surf_vertices[i*3 + 2] = tri[2];
        }
    }
};

struct MakeFEMObjFromFile : zeno::INode{
    virtual void apply() override {
        auto NodeFile = get_input<zeno::StringObject>("NodeFile")->get();
        auto EleFile = get_input<zeno::StringObject>("EleFile")->get();    

        VecXd vertices;
        VecXi elements;
        load_node_file(NodeFile.c_str(),vertices);
        load_ele_file(EleFile.c_str(),elements);

        auto res = std::make_shared<FEMMesh>(vertices,elements);

        auto resGeo =  std::make_shared<zeno::PrimitiveObject>();
        auto &pos = resGeo->add_attr<zeno::vec3f>("pos");

        for(int i=0;i<res->vertices.size()/3;++i)
            pos.emplace_back(vertices[i*3 + 0],vertices[i*3 + 1],vertices[i*3 + 2]);

        for(int i=0;i < res->elements.size() / 4;++i){
            auto tet = Vec4i(res->elements[i*4 + 0],
                             res->elements[i*4 + 1],
                             res->elements[i*4 + 2],
                             res->elements[i*4 + 3]);
            resGeo->tris.emplace_back(tet[0],tet[1],tet[2]);
            resGeo->tris.emplace_back(tet[1],tet[3],tet[2]);
            resGeo->tris.emplace_back(tet[0],tet[2],tet[3]);
            resGeo->tris.emplace_back(tet[0],tet[3],tet[1]);
        }

        pos.resize(res->vertices.size());
        set_output("FEMMesh",std::move(res));
        set_output("FEMMeshGeo", std::move(resGeo));
    }

    int load_node_file(const char* filename,VecXd& vertices){
        size_t num_vertices,space_dimension,d1,d2;
        std::ifstream fin;
        try {
            fin.open(filename);
            if (!fin.is_open()) {
                std::cerr << "ERROR::TET::FAILED::" << std::string(filename) << std::endl;
                return -1;
            }
            fin >> num_vertices >> space_dimension >> d1 >> d2;
            vertices.resize(num_vertices * space_dimension);

            for(size_t vert_id = 0;vert_id < num_vertices;++vert_id) {
                fin >> d1;
                for (size_t i = 0; i < space_dimension; ++i)
                    fin >> vertices[vert_id * space_dimension + i];
            }
            fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
            return -1;
        }
        return 0;
    }  

    int load_ele_file(const char* filename,VecXi& elements){
        size_t nm_elms,elm_size,v_start_idx,elm_idx;
        std::ifstream fin;
        size_t d2;
        try {
            fin.open(filename);
            if (!fin.is_open()) {
                std::cerr << "ERROR::TET::FAILED::" << std::string(filename) << std::endl;
                return -1;
            }
            fin >> nm_elms >> elm_size >> v_start_idx; 
            elements.resize(nm_elms * elm_size);

            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id) {
                fin >> elm_idx;
                for (size_t i = 0; i < elm_size; ++i) {
                    fin >> elements[elm_id * elm_size + i];
                    elements[elm_id * elm_size + i] -= v_start_idx;
                }
            }
            fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
            return -1;
        }
        return 0;
    }  
};

ZENDEFNODE(MakeFEMObjFromFile, {
    {{"readpath","NodeFile"},{"readpath", "EleFile"}},
    {"FEMMesh", "FEMMeshGeo"},
    {},
    {"FEM"},
});

struct AnisotropicForceModel : zeno::IObject{
    void ComputePhi(const Mat3x3d& Act,
            const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
            const VecXd& model_params,const Mat3x3d& F,FEM_Scaler& phi) const {
        Vec3d Is;
        Mat3x3d F_act = F * Act.inverse();
        ComputeAnisotrpicInvarients(fiber_direction,aniso_weight,F_act,Is);
        // std::cout << "ComputePhi Is " << Is.transpose() << "\n" << fiber_direction << "\n" << aniso_weight << "\n" << F_act << std::endl;
        FEM_Scaler I1_d = evalI1_delta(aniso_weight,fiber_direction,F_act);

        FEM_Scaler E = model_params[0];
        FEM_Scaler nu = model_params[1];
        FEM_Scaler lambda = Enu2Lambda(E,nu);
        FEM_Scaler mu = Enu2Mu(E,nu);

        phi = mu/2 * (Is[0] - I1_d) + lambda/2 * (Is[2] - 1) * (Is[2] - 1);
    }
    void ComputePhiDeriv(const Mat3x3d& Act,
            const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
            const VecXd& model_params, const Mat3x3d& F,FEM_Scaler &phi,Vec9d &dphi) const override{
        Vec3d Is;
        std::vector<Vec9d> Ds(3);

        Mat3x3d A_inv = Act.inverse();
        Mat3x3d F_act = F * A_inv;

        ComputeAnisotropicInvarientsDeriv(fiber_direction,aniso_weight,F_act,Is,Ds);
        // std::cout << "ComputePhiDeriv Is " << Is.transpose() << "\n" << fiber_direction << "\n" << aniso_weight << "\n" << F_act << std::endl;


        FEM_Scaler I1_d = evalI1_delta(aniso_weight,fiber_direction,F_act);
        Vec9d I1_d_deriv = MatHelper::VEC(evalI1_delta_deriv(aniso_weight,fiber_direction));

        FEM_Scaler E = model_params[0];
        FEM_Scaler nu = model_params[1];
        FEM_Scaler lambda = Enu2Lambda(E,nu);
        FEM_Scaler mu = Enu2Mu(E,nu);
        Mat9x9d dFactdF = EvaldFactdF(A_inv);

        phi = mu/2 * (Is[0] - I1_d) + lambda/2 * (Is[2] - 1) * (Is[2] - 1);
        dphi = mu/2 * (Ds[0] - I1_d_deriv) + lambda * (Is[2] - 1) * Ds[2];
        dphi = dFactdF.transpose() * dphi;
    }
    void ComputePhiDerivHessian(const Mat3x3d& Act,
            const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
            const VecXd& model_params,const Mat3x3d &F,FEM_Scaler& phi,Vec9d &dphi, Mat9x9d &ddphi,bool enforcing_spd = true) const override{
        Vec3d Is;
        std::vector<Vec9d> Ds(3);
        std::vector<Mat9x9d> Hs(3);

        Mat3x3d A_inv = Act.inverse();
        Mat3x3d F_act = F * A_inv;

        ComputeAnisotropicInvarientsDerivHessian(fiber_direction,aniso_weight,F_act,Is,Ds,Hs);

        FEM_Scaler I1_d = evalI1_delta(aniso_weight,fiber_direction,F_act);
        Vec9d I1_d_deriv = MatHelper::VEC(evalI1_delta_deriv(aniso_weight,fiber_direction));
        Mat9x9d dFactdF = EvaldFactdF(A_inv);

        FEM_Scaler E = model_params[0];
        FEM_Scaler nu = model_params[1];
        FEM_Scaler lambda = Enu2Lambda(E,nu);
        FEM_Scaler mu = Enu2Mu(E,nu);
        phi = mu/2 * (Is[0] - I1_d) + lambda/2 * (Is[2] - 1) * (Is[2] - 1);
        dphi = mu/2 * (Ds[0] - I1_d_deriv) + lambda * (Is[2] - 1) * Ds[2];
        dphi = dFactdF.transpose() * dphi;
        ddphi = mu/2 * Hs[0] + lambda * MatHelper::DYADIC(Ds[2],Ds[2]) + lambda * (Is[2] - 1) * Hs[2];
        ddphi = dFactdF.transpose() * ddphi * dFactdF;   
    }

};

}