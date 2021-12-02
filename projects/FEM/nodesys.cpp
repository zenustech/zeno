#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

// #include <anisotropic_NH.h>
#include <stable_anisotropic_NH.h>
#include <diriclet_damping.h>
#include <stable_isotropic_NH.h>
#include <bspline_isotropic_model.h>
#include <stable_Stvk.h>

#include <quasi_static_solver.h>
#include <backward_euler_integrator.h>
#include <fstream>
#include <algorithm>

#include <matrix_helper.hpp>
#include<Eigen/SparseCholesky>
#include <iomanip>

#include <cmath>

#include "matrixObject.h"

#include <cubicBspline.h>

#include <time.h>

namespace{

using namespace zeno;

struct FEMMesh : zeno::IObject{
    FEMMesh() = default;
    std::shared_ptr<PrimitiveObject> _mesh;
    std::vector<int> _bouDoFs;
    std::vector<int> _freeDoFs;
    std::vector<int> _DoF2FreeDoF;
    std::vector<int> _SpMatFreeDoFs;
    std::vector<Mat12x12i> _elmSpIndices;

    std::vector<double> _elmYoungModulus;
    std::vector<double> _elmPossonRatio;
    std::vector<double> _elmDamp;
    std::vector<double> _elmDensity;

    std::vector<Vec3d> _elmFiberDir;
    std::vector<Vec3d> _elmAct;


    std::vector<double> _elmMass;
    std::vector<double> _elmVolume;
    std::vector<Mat9x12d> _elmdFdx;
    std::vector<Mat4x4d> _elmMinv;


    SpMat _connMatrix;
    SpMat _freeConnMatrix;

    std::vector<int> _closeBindPoints;
    std::vector<int> _farBindPoints;

    Eigen::Map<const SpMat> MapHMatrix(const FEM_Scaler* HValBuffer){
        size_t n = _mesh->size() * 3;
        return Eigen::Map<const SpMat>(n,n,_connMatrix.nonZeros(),
            _connMatrix.outerIndexPtr(),_connMatrix.innerIndexPtr(),HValBuffer);
    }

    Eigen::Map<const SpMat> MapHucMatrix(const FEM_Scaler* HucValBuffer) {
        size_t nuc = _freeDoFs.size();
        return Eigen::Map<const SpMat>(nuc,nuc,_freeConnMatrix.nonZeros(),
            _freeConnMatrix.outerIndexPtr(),_freeConnMatrix.innerIndexPtr(),HucValBuffer);
    }

    Eigen::Map<SpMat> MapHMatrixRef(FEM_Scaler* HValBuffer){
        size_t n = _mesh->size() * 3;
        return Eigen::Map<SpMat>(n,n,_connMatrix.nonZeros(),
            _connMatrix.outerIndexPtr(),_connMatrix.innerIndexPtr(),HValBuffer);
    }

    Eigen::Map<SpMat> MapHucMatrixRef(FEM_Scaler* HucValBuffer) {
        size_t nuc = _freeDoFs.size();
        return Eigen::Map<SpMat>(nuc,nuc,_freeConnMatrix.nonZeros(),
            _freeConnMatrix.outerIndexPtr(),_freeConnMatrix.innerIndexPtr(),HucValBuffer);
    }

    void DoPreComputation() {
        size_t nm_elms = _mesh->quads.size();
        _elmVolume.resize(nm_elms);
        _elmdFdx.resize(nm_elms);
        _elmMass.resize(nm_elms);
        _elmMinv.resize(nm_elms);
        for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
            auto elm = _mesh->quads[elm_id];
            Mat4x4d M;
            for(size_t i = 0;i < 4;++i){
                auto vert = _mesh->verts[elm[i]];
                M.block(0,i,3,1) << vert[0],vert[1],vert[2];
            }
            M.bottomRows(1).setConstant(1.0);
            _elmVolume[elm_id] = fabs(M.determinant()) / 6;
            _elmMass[elm_id] = _elmVolume[elm_id] * _elmDensity[elm_id];


            Mat3x3d Dm;
            for(size_t i = 1;i < 4;++i){
                auto vert = _mesh->verts[elm[i]];
                auto vert0 = _mesh->verts[elm[0]];
                Dm.col(i - 1) << vert[0]-vert0[0],vert[1]-vert0[1],vert[2]-vert0[2];
            }

            Mat3x3d DmInv = Dm.inverse();
            _elmMinv[elm_id] = M.inverse();


            double m = DmInv(0,0);
            double n = DmInv(0,1);
            double o = DmInv(0,2);
            double p = DmInv(1,0);
            double q = DmInv(1,1);
            double r = DmInv(1,2);
            double s = DmInv(2,0);
            double t = DmInv(2,1);
            double u = DmInv(2,2);

            double t1 = - m - p - s;
            double t2 = - n - q - t;
            double t3 = - o - r - u; 

            _elmdFdx[elm_id] << 
                t1, 0, 0, m, 0, 0, p, 0, 0, s, 0, 0, 
                 0,t1, 0, 0, m, 0, 0, p, 0, 0, s, 0,
                 0, 0,t1, 0, 0, m, 0, 0, p, 0, 0, s,
                t2, 0, 0, n, 0, 0, q, 0, 0, t, 0, 0,
                 0,t2, 0, 0, n, 0, 0, q, 0, 0, t, 0,
                 0, 0,t2, 0, 0, n, 0, 0, q, 0, 0, t,
                t3, 0, 0, o, 0, 0, r, 0, 0, u, 0, 0,
                 0,t3, 0, 0, o, 0, 0, r, 0, 0, u, 0,
                 0, 0,t3, 0, 0, o, 0, 0, r, 0, 0, u;

        }
    }
// load .node file
    void LoadVerticesFromFile(const std::string& filename) {
        size_t num_vertices,space_dimension,d1,d2;
        std::ifstream node_fin;
        try {
            node_fin.open(filename.c_str());
            if (!node_fin.is_open()) {
                std::cerr << "ERROR::NODE::FAILED::" << filename << std::endl;
            }
            node_fin >> num_vertices >> space_dimension >> d1 >> d2;
            auto &pos = _mesh->add_attr<vec3f>("pos");
            pos.resize(num_vertices);

            for(size_t vert_id = 0;vert_id < num_vertices;++vert_id) {
                node_fin >> d1;
                for (size_t i = 0; i < space_dimension; ++i)
                    node_fin >> pos[vert_id][i];
            }
            node_fin.close();

            // std::cout << "OUT VERTICES : " << std::endl;
            // for(size_t i = 0;i < num_vertices;++i){
            //     std::cout << "P<" << i << "> :\t" << pos[i][0] << "\t" << pos[i][1] << "\t" << pos[i][2] << std::endl;
            // }
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }

    }

    void LoadBindingPoints(const std::string& bindfile) {
        size_t nm_closed_points,nm_far_points;
        std::ifstream bind_fin;
        try{
            bind_fin.open(bindfile.c_str());
            if (!bind_fin.is_open()) {
                std::cerr << "ERROR::NODE::FAILED::" << bindfile << std::endl;
            }
            bind_fin >> nm_closed_points >> nm_far_points;
            _closeBindPoints.resize(nm_closed_points);
            _farBindPoints.resize(nm_far_points);

            for(size_t i = 0;i < nm_closed_points;++i)
                bind_fin >> _closeBindPoints[i];

            for(size_t i = 0;i < nm_far_points;++i)
                bind_fin >> _farBindPoints[i];
            bind_fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }
    }

    void LoadElementsFromFile(const std::string& filename) {
        size_t nm_elms,elm_size,v_start_idx,elm_idx;
        std::ifstream ele_fin;
        try {
            ele_fin.open(filename.c_str());
            if (!ele_fin.is_open()) {
                std::cerr << "ERROR::TET::FAILED::" << filename << std::endl;
            }
            ele_fin >> nm_elms >> elm_size >> v_start_idx;

            auto& quads = _mesh->quads;
            quads.resize(nm_elms);

            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id) {
                ele_fin >> elm_idx;
                for (size_t i = 0; i < elm_size; ++i) {
                    ele_fin >> quads[elm_id][i];
                    quads[elm_id][i] -= v_start_idx;
                }

                // std::cout << "E<" << elm_idx << "> : \t" << quads[elm_id][0] << "\t" << quads[elm_id][1] << "\t" << quads[elm_id][2] << "\t" << quads[elm_id][3] << std::endl;

            }
            ele_fin.close();

            // for(size_t i = 0;i < nm_elms;++i){
            //     std::cout << "E<" << i << "> : \t" << quads[i][0] << "\t" << quads[i][1] << "\t" << quads[i][2] << "\t" << quads[i][3] << std::endl;
            // }
            // std::cout << "HEADER : " << nm_elms << "\t" << elm_size << "\t" << v_start_idx << "\t" << quads.size() << std::endl;
            // throw std::runtime_error("ELE_INPUT_CHECK");

        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }
    }

    void LoadBoundaryIndicesFromFile(const std::string& filename) {
        size_t nm_cons_dofs,start_idx;
        std::ifstream bou_fin;
        try{
            bou_fin.open(filename.c_str());
            if(!bou_fin.is_open()){
                std::cerr << "ERROR::BOU::FAILED::" << filename << std::endl;
            }
            bou_fin >> nm_cons_dofs >> start_idx;
            _bouDoFs.resize(nm_cons_dofs);
            for(size_t c_id = 0;c_id < nm_cons_dofs;++c_id){
                bou_fin >> _bouDoFs[c_id];
                _bouDoFs[c_id] -= start_idx;
            }
            bou_fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }
    }

    void LoadBoundaryVerticesFromFile(const std::string& filename){
        size_t nm_con_vertices,start_idx;
        std::ifstream bou_fin;
        try{
            bou_fin.open(filename.c_str());
            if(!bou_fin.is_open()){
                std::cerr << "ERROR::BOU::FAILED::" << filename << std::endl;
            }
            bou_fin >> nm_con_vertices >> start_idx;
            _bouDoFs.resize(nm_con_vertices * 3);
            for(size_t i = 0;i < nm_con_vertices;++i){
                size_t vert_idx;
                bou_fin >> vert_idx;
                vert_idx -= start_idx;
                // std::cout << "vert_idx ; " << vert_idx << std::endl;
                _bouDoFs[i*3 + 0] = vert_idx * 3 + 0;
                _bouDoFs[i*3 + 1] = vert_idx * 3 + 1;
                _bouDoFs[i*3 + 2] = vert_idx * 3 + 2;
            }

            std::sort(_bouDoFs.begin(),_bouDoFs.end(),std::greater<int>());

            bou_fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }
    }

    void UpdateDoFsMapping() {
        int nmDoFs = _mesh->verts.size() * 3;
        int nmBouDoFs = _bouDoFs.size();
        int nmFreeDoFs = nmDoFs - _bouDoFs.size();
        std::cout << "nmFree : " << nmFreeDoFs << "\t" << nmDoFs << "\t" << _bouDoFs.size() << std::endl;
        _freeDoFs.resize(nmFreeDoFs);


        for(size_t cdof_idx = 0,dof = 0,ucdof_count = 0;dof < nmDoFs;++dof){
            if(cdof_idx >= nmBouDoFs || dof != _bouDoFs[cdof_idx]){
                // std::cout << "cdof_idx  = " << cdof_idx << "\t" << 
                _freeDoFs[ucdof_count] = dof;
                ++ucdof_count;
            }
            else
                ++cdof_idx;
        }


        _DoF2FreeDoF.resize(nmDoFs);
        // std::cout << "nmDoFs : " << nmDoFs << std::endl;
        std::fill(_DoF2FreeDoF.begin(),_DoF2FreeDoF.end(),-1);
        // std::cout << "..." << std::endl;
        for(size_t i = 0;i < _freeDoFs.size();++i){
            int ucdof = _freeDoFs[i];
            _DoF2FreeDoF[ucdof] = i;
        }

        size_t nm_elms = _mesh->quads.size();
        std::set<Triplet,triplet_cmp> connTriplets;
        size_t nm_insertions = 0;
        for (size_t elm_id = 0; elm_id < nm_elms; ++elm_id) {
            const auto& elm = _mesh->quads[elm_id];
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    for (size_t k = 0; k < 3; ++k)
                        for (size_t l = 0; l < 3; ++l) {
                            size_t row = elm[i] * 3 + k;
                            size_t col = elm[j] * 3 + l;
                            if(row > col)
                                continue;
                            if(row == col){
                                if(row > nmDoFs || col > nmDoFs){
                                    std::cout << "error warning : " << row << "\t" << col << "\t" << nmDoFs << std::endl;
                                    std::cout << "nm_elms : " << nm_elms << std::endl;
                                    std::cout << "elm : " << elm[0] << "\t" << elm[1] << "\t" << elm[2] << "\t" << elm[3] << std::endl;
                                    throw std::runtime_error("invalid triplet");
                                }
                                connTriplets.insert(Triplet(row, col, 1.0));
                                ++nm_insertions;
                            }else{
                                connTriplets.insert(Triplet(row, col, 1.0));
                                connTriplets.insert(Triplet(col, row, 1.0));
                                nm_insertions += 2;
                            }
                        }
        }
        _connMatrix = SpMat(nmDoFs,nmDoFs);

        _connMatrix.setFromTriplets(connTriplets.begin(),connTriplets.end());
        
        _connMatrix.makeCompressed();

        std::set<Triplet,triplet_cmp> freeConnTriplets;
        nm_insertions = 0;
        for(size_t elm_id = 0;elm_id < nm_elms;++elm_id) {
            const auto& elm = _mesh->quads[elm_id];
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    for (size_t k = 0; k < 3; ++k)
                        for (size_t l = 0; l < 3; ++l) {
                            size_t row = _DoF2FreeDoF[elm[i] * 3 + k];
                            size_t col = _DoF2FreeDoF[elm[j] * 3 + l];
                            if(row == -1 || col == -1 || row > col)
                                continue;
                            if(row == col){
                                freeConnTriplets.insert(Triplet(row,col,1.0));
                                nm_insertions++;
                            }else{
                                freeConnTriplets.insert(Triplet(row,col,1.0));
                                freeConnTriplets.insert(Triplet(col,row,1.0));
                                nm_insertions += 2;
                            }
                        }
        }
        _freeConnMatrix = SpMat(nmFreeDoFs,nmFreeDoFs);
        _freeConnMatrix.setFromTriplets(freeConnTriplets.begin(),freeConnTriplets.end());
        _freeConnMatrix.makeCompressed();

        _SpMatFreeDoFs.resize(_freeConnMatrix.nonZeros());
        size_t uc_idx = 0;
        size_t idx = 0;
        for(size_t k = 0;k < size_t(_connMatrix.outerSize());++k)
            for(SpMat::InnerIterator it(_connMatrix,k);it;++it){
                size_t row = it.row();
                size_t col = it.col();
                if(_DoF2FreeDoF[row] == -1 || _DoF2FreeDoF[col] == -1){
                    idx++;
                    continue;
                }
                _SpMatFreeDoFs[uc_idx] = idx;
                ++uc_idx;
                ++idx;
            }
    }
};

struct GetFEMMaterial : zeno::INode {
    virtual void apply() override {
        auto femmesh = get_input<FEMMesh>("femmesh");
        auto res = std::make_shared<PrimitiveObject>();

        size_t nm_elms = femmesh->_mesh->quads.size();

        auto& Es = res->add_attr<float>("E");
        auto& nus = res->add_attr<float>("nu");
        auto& pos = res->add_attr<zeno::vec3f>("pos");
        auto& phi = res->add_attr<float>("phi");
        auto& fd = res->add_attr<zeno::vec3f>("fd");

        res->resize(nm_elms);

        for(size_t i = 0;i < nm_elms;++i){
            Es[i] = femmesh->_elmYoungModulus[i];
            nus[i] = femmesh->_elmPossonRatio[i];
            phi[i] = femmesh->_elmDensity[i];

            const auto& tet = femmesh->_mesh->quads[i];
            pos[i] = zeno::vec3f(0);
            for(size_t j = 0;j < 4;++j)
                pos[i] += femmesh->_mesh->verts[tet[j]]/4;

            fd[i] = zeno::vec3f(femmesh->_elmFiberDir[i][0],femmesh->_elmFiberDir[i][1],femmesh->_elmFiberDir[i][2]);
        }

        set_output("mprim",std::move(res));
    }
};
ZENDEFNODE(GetFEMMaterial, {
    {"femmesh"},
    {"mprim"},
    {},
    {"FEM"},
});


struct FEMMeshToPrimitive : zeno::INode{
    virtual void apply() override {
        auto res = get_input<FEMMesh>("femmesh");
        set_output("primitive", res->_mesh);
    }
};


ZENDEFNODE(FEMMeshToPrimitive, {
    {"femmesh"},
    {"primitive"},
    {},
    {"FEM"},
});

// struct FiberFieldToPrimtive : zeno::INode {
//     virtual void apply() override {
//         auto fp = get_input<zeno::PrimitiveObject>("fp");
//         auto prim = get_input<zeno::PrimitiveObject>("prim");

//         float sigma = 2;


//         #pragma omp parallel for
//         for(size_t i = 0;i < prim->size();++i){
//             fElmID[elm_id] = float(elm_id);
//             auto tet = tets[elm_id];
//             fpos[elm_id] = zeno::vec3f(0.0,0.0,0.0);
//             for(size_t i = 0;i < 4;++i){
//                 fpos[elm_id] += mpos[tet[i]];
//             }
//             fpos[elm_id] /= 4;

//             auto& fdir = forient[elm_id];
//             fdir = zeno::vec3f(0);
//             for(size_t i = 0;i < fp->size();++i){
//                 const auto& ppos = fp->verts[i];
//                 const auto& pdir = fp->attr<zeno::vec3f>("vel")[i];

//                 float dissqrt = zeno::lengthSquared(fpos[elm_id] - ppos);
//                 float weight = exp(-dissqrt / pow(sigma,2));

//                 fdir += pdir * weight;
//             }
//             fdir /= zeno::length(fdir);

//             fact[elm_id] = zeno::vec3f(1.0);
//         }

//         set_output("fiberOut",fiber);     
//     }
// };

// ZENDEFNODE(FiberFieldToPrimtive, {
//     {"fp","prim"},
//     {"fiberOut"},
//     {},
//     {"FEM"}
// });

struct ParticlesToSegments : zeno::INode {
    virtual void apply() override {
        auto particles = get_input<zeno::PrimitiveObject>("particles");
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto attr_name = get_param<std::string>("attr_name");
        const auto& ppos = particles->verts;
        const auto& pvel = particles->attr<zeno::vec3f>(attr_name);

        auto segs = std::make_shared<zeno::PrimitiveObject>();
        segs->resize(particles->size() * 2);
        auto& segLines = segs->lines;
        segLines.resize(particles->size());
        auto& spos = segs->verts;

        for(size_t i = 0;i < particles->size();++i){
            segLines[i] = zeno::vec2i(i,i + particles->size());
            spos[i] = ppos[i];
            spos[i + particles->size()] = spos[i] + dt * pvel[i];
        }

        set_output("seg",segs);
    }
};

ZENDEFNODE(ParticlesToSegments, {
    {"particles","dt"},
    {"seg"},
    {{"string","attr_name",""}},
    {"FEM"},
});


struct DeformFiberWithFE : zeno::INode {
    virtual void apply() override {
        auto rfiber = get_input<zeno::PrimitiveObject>("restFibers");

        const auto& restm = get_input<FEMMesh>("restShape");
        const auto& deform = get_input<zeno::PrimitiveObject>("deformedShape");

        const auto& elmIDs = rfiber->attr<float>("elmID");

        auto dfiber = std::make_shared<zeno::PrimitiveObject>();
        dfiber->add_attr<float>("elmID");
        dfiber->add_attr<zeno::vec3f>("vel");
        dfiber->resize(rfiber->size());
        dfiber->lines = rfiber->lines;

        auto& fverts = dfiber->verts;
        auto& fdirs = dfiber->attr<zeno::vec3f>("vel"); 
        for(size_t fid = 0;fid < dfiber->size();++fid){
            size_t elm_id = size_t(elmIDs[fid]);
            const auto& tet = restm->_mesh->quads[elm_id];
            fverts[fid] = zeno::vec3f(0);
            for(size_t i = 0;i < 4;++i){
                fverts[fid] += deform->verts[tet[i]];
            }
            fverts[fid] /= 4;

            // compute the deformation gradient
            Mat4x4d G;
            for(size_t i = 0;i < 4;++i){
                G.col(i) << deform->verts[tet[i]][0],deform->verts[tet[i]][1],deform->verts[tet[i]][2],1.0;
            }
            G = G * restm->_elmMinv[elm_id];
            auto F = G.topLeftCorner(3,3);  

            auto rdir = rfiber->attr<zeno::vec3f>("vel")[fid];
            Vec3d dir;
            dir << rdir[0],rdir[1],rdir[2];
            dir = F * dir;
            fdirs[fid] = zeno::vec3f(dir[0],dir[1],dir[2]);
        }

        set_output("deformedFiber",std::move(dfiber));
    }
};

ZENDEFNODE(DeformFiberWithFE, {
    {"restFibers","restShape","deformedShape"},
    {"deformedFiber"},
    {},
    {"FEM"},
});

struct MakeFEMGeoFromFile : zeno::INode {
    virtual void apply() override {
        auto node_file = get_input<zeno::StringObject>("NodeFile")->get();
        auto ele_file = get_input<zeno::StringObject>("EleFile")->get();
        auto res = std::make_shared<PrimitiveObject>();    

        auto &pos = res->add_attr<vec3f>("pos");
        auto &vol = res->add_attr<float>("vol");
        auto &cm = res->add_attr<float>("cm");

        size_t num_vertices,space_dimension,d1,d2;
        std::ifstream node_fin;
        try {
            node_fin.open(node_file.c_str());
            if (!node_fin.is_open()) {
                std::cerr << "ERROR::NODE::FAILED::" << node_file << std::endl;
            }
            node_fin >> num_vertices >> space_dimension >> d1 >> d2;
            pos.resize(num_vertices);

            for(size_t vert_id = 0;vert_id < num_vertices;++vert_id) {
                node_fin >> d1;
                for (size_t i = 0; i < space_dimension; ++i)
                    node_fin >> pos[vert_id][i];
            }
            node_fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }

        auto& quads = res->quads;

        size_t nm_elms,elm_size,v_start_idx,elm_idx;
        std::ifstream ele_fin;
        try {
            ele_fin.open(ele_file.c_str());
            if (!ele_fin.is_open()) {
                std::cerr << "ERROR::TET::FAILED::" << ele_file << std::endl;
            }
            ele_fin >> nm_elms >> elm_size >> v_start_idx;
            quads.resize(nm_elms);
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id) {
                ele_fin >> elm_idx;
                for (size_t i = 0; i < elm_size; ++i) {
                    ele_fin >> quads[elm_id][i];
                    quads[elm_id][i] -= v_start_idx;
                }
            }
            ele_fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }
        res->resize(num_vertices);
        vol.resize(num_vertices,0);
        cm.resize(num_vertices,0);

        for(size_t i = 0;i < res->quads.size();++i){
            auto tet = res->quads[i];
            res->tris.emplace_back(tet[0],tet[1],tet[2]);
            res->tris.emplace_back(tet[1],tet[3],tet[2]);
            res->tris.emplace_back(tet[0],tet[2],tet[3]);
            res->tris.emplace_back(tet[0],tet[3],tet[1]);
        }
//      Compute Characteristic Gradient
        std::vector<FEM_Scaler> vert_incident_volume;
        std::vector<FEM_Scaler> vert_one_ring_surface;
        vert_incident_volume.resize(num_vertices,0);
        vert_one_ring_surface.resize(num_vertices,0);

        for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
            const auto& tet = res->quads[elm_id];
            Vec3d v0,v1,v2,v3;
            v0 << pos[tet[0]][0],pos[tet[0]][1],pos[tet[0]][2],pos[tet[0]][3];
            v1 << pos[tet[1]][0],pos[tet[1]][1],pos[tet[1]][2],pos[tet[1]][3];
            v2 << pos[tet[2]][0],pos[tet[2]][1],pos[tet[2]][2],pos[tet[2]][3];
            v3 << pos[tet[3]][0],pos[tet[3]][1],pos[tet[3]][2],pos[tet[3]][3];

            Mat4x4d tetV;
            tetV.col(0) << v0,1.0;
            tetV.col(1) << v1,1.0;
            tetV.col(2) << v2,1.0;
            tetV.col(3) << v3,1.0;

            FEM_Scaler tvol = fabs(tetV.determinant()) / 6;

            vol[tet[0]] = vol[tet[0]] + tvol/4;
            vol[tet[1]] = vol[tet[1]] + tvol/4;
            vol[tet[2]] = vol[tet[2]] + tvol/4;
            vol[tet[3]] = vol[tet[3]] + tvol/4;

            vert_incident_volume[tet[0]] += tvol;
            vert_incident_volume[tet[1]] += tvol;
            vert_incident_volume[tet[2]] += tvol;
            vert_incident_volume[tet[3]] += tvol;

            vert_one_ring_surface[tet[0]] += tvol / MatHelper::Height(v1,v2,v3,v0);
            vert_one_ring_surface[tet[1]] += tvol / MatHelper::Height(v2,v3,v0,v1);
            vert_one_ring_surface[tet[2]] += tvol / MatHelper::Height(v3,v0,v1,v2);
            vert_one_ring_surface[tet[3]] += tvol / MatHelper::Height(v0,v1,v2,v3);
        }

        for(size_t i = 0;i < num_vertices;++i){
            cm[i] = vert_one_ring_surface[i];
        }

        // std::cout << "OUTPUT INCIDENT ATTRBS : " << std::endl;
        // for(size_t i = 0;i < num_vertices;++i){
        //     // std::cout << "VERT<" << i << "> : \t" << cm[i] << "\t" << vol[i] << std::endl;
        // }

        set_output("geo",std::move(res));
    }
};

ZENDEFNODE(MakeFEMGeoFromFile, {
    {{"readpath","NodeFile"},{"readpath", "EleFile"}},
    {"geo"},
    {},
    {"FEM"},
});

struct PrimitiveToFEMMesh : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("femGeo");
        if(prim->quads.size() <= 0)
            throw std::runtime_error("PRIM HAS NO TOPOLOGY INFORMATION");

        auto res = std::make_shared<FEMMesh>();

        float uni_phi = get_param<float>("phi");
        float uni_E = get_param<float>("E");
        float uni_nu = get_param<float>("nu");
        float uni_d = get_param<float>("dampingCoeff");
        // zeno::vec3f default_fdir = get_param<zeno::vec3f>("fiberDir");

        zeno::vec3f default_fdir = zeno::vec3f(1,0,0);
        size_t nm_elms = prim->quads.size();

        res->_mesh = prim;
        res->_elmYoungModulus.resize(nm_elms);
        res->_elmPossonRatio.resize(nm_elms);
        res->_elmDamp.resize(nm_elms);
        res->_elmDensity.resize(nm_elms);
        res->_elmFiberDir.resize(nm_elms);

        // Set material properties
        for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
            const auto& tet = prim->quads[elm_id];
            res->_elmDensity[elm_id] = 0;
            res->_elmYoungModulus[elm_id] = 0;
            res->_elmPossonRatio[elm_id] = 0;
            res->_elmDamp[elm_id] = 0;
            for(size_t i = 0;i < 4;++i){
                size_t idx = tet[i];
                res->_elmDensity[elm_id] += prim->has_attr("phi") ? prim->attr<float>("phi")[idx] : uni_phi;
                res->_elmYoungModulus[elm_id] += prim->has_attr("E") ? prim->attr<float>("E")[idx] : uni_E;
                res->_elmPossonRatio[elm_id] += prim->has_attr("nu") ? prim->attr<float>("nu")[idx] : uni_nu;
                res->_elmDamp[elm_id] += prim->has_attr("dampingCoeff") ? prim->attr<float>("dampingCoeff")[idx] : uni_d;
                zeno::vec3f fdir = prim->has_attr("fiberDir") ? prim->attr<zeno::vec3f>("fiberDir")[idx] : default_fdir;
                Vec3d fdir_vec = Vec3d(fdir[0],fdir[1],fdir[2]);
                if(fdir_vec.norm() > 1e-5)
                    fdir_vec /= fdir_vec.norm();

                res->_elmFiberDir[elm_id] += fdir_vec;

                // if(elm_id == 8640){
                //     std::cout << "PARTICLE_TO_FEM_ASSEMBLE: "<< std::endl;
                //     std::cout << "FDIR<" << i << "> : " << fdir[0] << "\t" << fdir[1] << "\t" << fdir[2] << std::endl;
                //     std::cout << "PARTICLE_TO_FEM_ASSEMBLE_END "<< std::endl;
                // }
            }
            res->_elmDensity[elm_id] /= 4;
            res->_elmYoungModulus[elm_id] /= 4;
            res->_elmPossonRatio[elm_id] /= 4;
            res->_elmDamp[elm_id] /= 4;   
            res->_elmFiberDir[elm_id] /= res->_elmFiberDir[elm_id].norm();  

            // if(elm_id == 8640){
            //     std::cout << "PARTICLE_TO_FEM: "<< std::endl;
            //     std::cout << res->_elmFiberDir[elm_id].transpose() << std::endl;
            //     std::cout << "PARTICLE_TO_FEM_END "<< std::endl;
            // }
        }

        // std::cout << "CHECK_MATERIAL: " << std::endl;
        // for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
        //     std::cout << "ELM<" << elm_id << ">:\t" << res->_elmYoungModulus[elm_id] << "\t" << res->_elmPossonRatio[elm_id] << "\t" << res->_elmDamp[elm_id] << 
        //         res->_elmFiberDir[elm_id].transpose() << std::endl;
        //     if(std::isnan(res->_elmFiberDir[elm_id].norm())){
        //         std::cout << "NAN VALUE DETECTED:" << elm_id << "\t" << res->_elmFiberDir[elm_id].transpose() << std::endl;
        //         throw std::runtime_error("NAN");
        //     }
        // }

        // throw std::runtime_error("MAT_CHECK");

        // Set the fixed points and fixed dofs
        res->_closeBindPoints.clear();
        res->_farBindPoints.clear();
        if(prim->has_attr("btag")){
            // std::cout << "PRIM HAS BTAG : " << std::endl;
            const auto& btags = prim->attr<float>("btag");

            // for(size_t i = 0;i < prim->size();++i){
            //     std::cout << "PBTAG<" << i << ">:\t" << btags[i] << std::endl;
            // }

            for(size_t i = 0;i < prim->size();++i){
                if(btags[i] == 1.0)
                    res->_closeBindPoints.emplace_back(i);
                if(btags[i] == 2.0)
                    res->_farBindPoints.emplace_back(i);
            }
        }
        // std::cout << "NM_CLOSE_BIND : \t" << res->_closeBindPoints.size() << std::endl;
        // std::cout << "NM_FAR_BIND : \t" << res->_farBindPoints.size() << std::endl;

        res->_bouDoFs.clear();
        for(size_t i = 0;i < res->_closeBindPoints.size();++i){
            size_t vert_idx = res->_closeBindPoints[i];
            res->_bouDoFs.emplace_back(vert_idx * 3 + 0);
            res->_bouDoFs.emplace_back(vert_idx * 3 + 1);
            res->_bouDoFs.emplace_back(vert_idx * 3 + 2);
        }
        for(size_t i = 0;i < res->_farBindPoints.size();++i){
            size_t vert_idx = res->_farBindPoints[i];
            res->_bouDoFs.emplace_back(vert_idx * 3 + 0);
            res->_bouDoFs.emplace_back(vert_idx * 3 + 1);
            res->_bouDoFs.emplace_back(vert_idx * 3 + 2);
        }
        std::sort(res->_bouDoFs.begin(),res->_bouDoFs.end(),std::less<int>());


        // throw std::runtime_error("CHECK BOU");

        res->UpdateDoFsMapping();
        res->DoPreComputation();

        set_output("fem",std::move(res));
    }
};

ZENDEFNODE(PrimitiveToFEMMesh, {
    {"femGeo"},
    {"fem"},
    {{"float","phi","1000"},{"float","E","1e6"},{"float","nu","0.45"},{"float","dampingCoeff","0"}},
    {"FEM"},
});

struct DeformPrimitiveWithFE : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<zeno::PrimitiveObject>("bindMesh");
        auto prim = get_input<zeno::PrimitiveObject>("prim");

        const auto& bindElmIDs = prim->attr<float>("elmID");
        const auto& interpWeight = prim->attr<zeno::vec3f>("interp_weight");

        for(size_t i = 0;i < prim->size();++i){
            size_t elm_id = int(bindElmIDs[i]);
            auto& vert = prim->verts[i];

            const auto& tet = mesh->quads[elm_id];
            Vec4d weight = Vec4d(
                interpWeight[i][0],
                interpWeight[i][1],
                interpWeight[i][2],
                1 - interpWeight[i][0] - interpWeight[i][1] - interpWeight[i][2]);

            prim->verts[i] = zeno::vec3f(0);
            for(size_t j = 0;j < 4;++j){
                const auto& int_vert = mesh->verts[tet[j]];
                prim->verts[i] += weight[j] * int_vert;
            }
        }
        set_output("primOut",std::move(prim));
    }
};

ZENDEFNODE(DeformPrimitiveWithFE, {
    {{"bindMesh"},{"prim"}},
    {"primOut"},
    {},
    {"FEM"},
});

struct TransformFEMMesh : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("inputMesh");
        zeno::vec3f translate = zeno::vec3f(0.0);
        if(has_input("translate")){
            translate = get_input<zeno::NumericObject>("translate")->get<zeno::vec3f>();
        }
        zeno::vec3f scale = zeno::vec3f(1.0);
        if(has_input("scale")){
            scale = get_input<zeno::NumericObject>("scale")->get<zeno::vec3f>();
        }

        for(size_t i = 0;i < mesh->_mesh->size();++i){
            for(size_t j = 0;j < 3;++j){
                mesh->_mesh->verts[i][j] += translate[j];
                mesh->_mesh->verts[i][j] *= scale[j];
            }
        }
        mesh->DoPreComputation();

        set_output("outMesh",mesh);
    }
};

ZENDEFNODE(TransformFEMMesh,{
    {{"inputMesh"},{"translate"},{"scale"}},
    {"outMesh"},
    {},
    {"FEM"}
});


struct MuscleModelObject : zeno::IObject {
    MuscleModelObject() = default;
    std::shared_ptr<BaseForceModel> _forceModel;
};

struct DampingForceModel : zeno::IObject {
    DampingForceModel() = default;
    std::shared_ptr<DiricletDampingModel> _dampForce;
};

struct MakeElasticForceModel : zeno::INode {
    virtual void apply() override {
        auto model_type = std::get<std::string>(get_param("ForceModel"));
        auto aniso_strength = get_param<float>("aniso_strength");
        auto res = std::make_shared<MuscleModelObject>();
        if(model_type == "Fiberic"){
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableAnisotropicMuscle(aniso_strength));
            // std::cout << "The Anisotropic Model is not stable yet" << std::endl;
            // throw std::runtime_error("The Anisotropic Model is not stable yet");
        }
        else if(model_type == "HyperElastic")
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableIsotropicMuscle());
        else if(model_type == "BSplineModel"){
            FEM_Scaler default_E = 1e7;
            FEM_Scaler default_nu = 0.499;
            res->_forceModel = std::shared_ptr<BaseForceModel>(new BSplineIsotropicMuscle(default_E,default_nu));
        }else if(model_type == "Stvk"){
            // std::cout << "LOADING STVK MODEL" << std::endl;
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableStvk());
        }
        else{
            std::cerr << "UNKNOWN MODEL_TYPE" << std::endl;
            throw std::runtime_error("UNKNOWN MODEL_TYPE");
        }
        set_output("BaseForceModel",res);
    }
};

ZENDEFNODE(MakeElasticForceModel, {
    {},
    {"BaseForceModel"},
    {{"enum HyperElastic Fiberic BSplineModel Stvk", "ForceModel", "HyperElastic"},{"float","aniso_strength","20"}},
    {"FEM"},
});


struct MakePlasticForceModel : zeno::INode {
    virtual void apply() override {
        auto model_type = std::get<std::string>(get_param("ForceModel"));
        auto aniso_strength = get_param<float>("aniso_strength");


        std::shared_ptr<ElasticModel> elastic_model_ptr;

        if(model_type == "Fiberic"){
            elastic_model_ptr = std::make_shared<StableAnisotropicMuscle>(aniso_strength);
        }
        else if(model_type == "HyperElastic")
            elastic_model_ptr = std::make_shared<StableIsotropicMuscle>();
        else if(model_type == "BSplineModel"){
            FEM_Scaler default_E = 1e7;
            FEM_Scaler default_nu = 0.499;
            elastic_model_ptr = std::make_shared<BSplineIsotropicMuscle>(default_E,default_nu);
        }else if(model_type == "Stvk"){
            elastic_model_ptr = std::make_shared<StableStvk>();
        }
        else{
            std::cerr << "UNKNOWN MODEL_TYPE" << std::endl;
            throw std::runtime_error("UNKNOWN MODEL_TYPE");
        }
        auto res = std::make_shared<MuscleModelObject>();
        res->_forceModel = std::make_shared<PlasticForceModel>(elastic_model_ptr);

        set_output("BaseForceModel",res);
    }
};

ZENDEFNODE(MakePlasticForceModel, {
    {},
    {"BaseForceModel"},
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

struct FEMIntegrator : zeno::IObject {
    FEMIntegrator() = default;
    std::shared_ptr<BaseIntegrator> _intPtr;
    std::vector<VecXd> _traj;
    size_t _stepID;
    VecXd _extForce;

    std::vector<FEM_Scaler> _objBuffer;
    std::vector<Vec12d> _derivBuffer;
    std::vector<Mat12x12d> _HBuffer;

    std::vector<Vec3d> kinematic_hardening_shifts;
    std::vector<Vec3d> plastic_strains;
    std::vector<Mat3x3d> PSs;
    std::vector<Vec3d> init_stresses;
    std::vector<Vec3d> init_strains;
    std::vector<FEM_Scaler> kinimatic_hardening_coeffs;
    std::vector<FEM_Scaler> yielding_stresses;

    std::vector<FEM_Scaler> restoring_strains;
    std::vector<FEM_Scaler> fail_strains;

    std::vector<bool> failed;
    std::vector<Vec3d> the_strains_failed;
    std::vector<Mat3x3d> Fs_failed;

    std::vector<zeno::vec3f> respStress_traj;
    std::vector<float> strain_traj;
    std::vector<FEM_Scaler> vms;
    std::vector<FEM_Scaler> rms;

    Vec3d uniformAct;

    VecXd& GetCurrentFrame() {return _traj[(_stepID + _intPtr->GetCouplingLength()) % _intPtr->GetCouplingLength()];} 

    void AssignElmAttribs(size_t elm_id,TetAttributes& attrbs,const std::shared_ptr<FEMMesh>& mesh,const Vec12d& ext_force) const {
        attrbs._elmID = elm_id;
        attrbs._Minv = mesh->_elmMinv[elm_id];
        attrbs._dFdX = mesh->_elmdFdx[elm_id];

        attrbs.emp.forient = mesh->_elmFiberDir[elm_id];
        Mat3x3d R = MatHelper::Orient2R(attrbs.emp.forient);
        attrbs.emp.Act = R * uniformAct.asDiagonal() * R.transpose();

        attrbs.emp.E = mesh->_elmYoungModulus[elm_id];
        attrbs.emp.nu = mesh->_elmPossonRatio[elm_id];
        attrbs.v = mesh->_elmDamp[elm_id];
        attrbs._volume = mesh->_elmVolume[elm_id];
        attrbs._density = mesh->_elmDensity[elm_id];
        attrbs._ext_f = ext_force; 

        attrbs.pmp.init_stress = init_stresses[elm_id];
        attrbs.pmp.init_strain = init_strains[elm_id];
        attrbs.pmp.isotropic_hardening_coeff = 0;
        attrbs.pmp.kinematic_hardening_coeff = kinimatic_hardening_coeffs[elm_id];
        attrbs.pmp.kinematic_hardening_shift = kinematic_hardening_shifts[elm_id];
        attrbs.pmp.plastic_strain = plastic_strains[elm_id];
        attrbs.pmp.PS = PSs[elm_id];
        attrbs.pmp.yield_stress = yielding_stresses[elm_id];
        attrbs.pmp.restoring_strain = restoring_strains[elm_id];
        attrbs.pmp.failed_strain = fail_strains[elm_id];
        attrbs.pmp.failed = failed[elm_id];
        attrbs.pmp.the_strain_failed = the_strains_failed[elm_id];
        attrbs.pmp.F_failed = Fs_failed[elm_id];
    }


    FEM_Scaler EvalObj(const std::shared_ptr<FEMMesh>& mesh,
        const std::shared_ptr<MuscleModelObject>& muscle,
        const std::shared_ptr<DampingForceModel>& damp) {
            FEM_Scaler obj = 0;
            size_t nm_elms = mesh->_mesh->quads.size();

            _objBuffer.resize(nm_elms);
            
            size_t clen = _intPtr->GetCouplingLength();

            #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                auto tet = mesh->_mesh->quads[elm_id];

                Vec12d elm_ext_force;
                RetrieveElmVector(tet,elm_ext_force,_extForce);
                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh,elm_ext_force);

                std::vector<Vec12d> elm_traj(clen);
                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],_traj[frameID]);
                }

                FEM_Scaler elm_obj = 0;
                _intPtr->EvalElmObj(attrbs,
                    muscle->_forceModel,
                    damp->_dampForce,
                    elm_traj,&_objBuffer[elm_id]);
            }

            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                obj += _objBuffer[elm_id];
            }

            return obj; 
    }

    FEM_Scaler EvalObjDeriv(const std::shared_ptr<FEMMesh>& mesh,
        const std::shared_ptr<MuscleModelObject>& muscle,
        const std::shared_ptr<DampingForceModel>& damp,
        VecXd& deriv) {
            FEM_Scaler obj = 0;
            size_t nm_elms = mesh->_mesh->quads.size();

            size_t clen = _intPtr->GetCouplingLength();

            _objBuffer.resize(nm_elms);
            _derivBuffer.resize(nm_elms);
            
            #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];

                Vec12d elm_ext_force;
                RetrieveElmVector(tet,elm_ext_force,_extForce);

                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh,elm_ext_force);

                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],_traj[frameID]);
                }

                FEM_Scaler elm_obj = 0;
                _intPtr->EvalElmObjDeriv(attrbs,
                    muscle->_forceModel,
                    damp->_dampForce,
                    elm_traj,&_objBuffer[elm_id],_derivBuffer[elm_id]);
            }

            deriv.setZero();
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                auto tet = mesh->_mesh->quads[elm_id];
                obj += _objBuffer[elm_id];
                AssembleElmVector(tet,_derivBuffer[elm_id],deriv);
            }

            return obj;
    }

    FEM_Scaler EvalObjDerivHessian(const std::shared_ptr<FEMMesh>& mesh,
        const std::shared_ptr<MuscleModelObject>& muscle,
        const std::shared_ptr<DampingForceModel>& damp,
        VecXd& deriv,
        VecXd& HValBuffer,
        bool enforce_spd) {
            FEM_Scaler obj = 0;

            size_t clen = _intPtr->GetCouplingLength();

            size_t nm_elms = mesh->_mesh->quads.size();

            _objBuffer.resize(nm_elms);
            _derivBuffer.resize(nm_elms);
            _HBuffer.resize(nm_elms);

            #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                // throw std::runtime_error("CHECK ATTRS");
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];

                Vec12d elm_ext_force;
                RetrieveElmVector(tet,elm_ext_force,_extForce);

                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh,elm_ext_force);         

                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],_traj[frameID]);
                }

                _intPtr->EvalElmObjDerivJacobi(attrbs,
                    muscle->_forceModel,
                    damp->_dampForce,
                    elm_traj,
                    &_objBuffer[elm_id],_derivBuffer[elm_id],_HBuffer[elm_id],enforce_spd);


                bool debug_int = false;
                if(debug_int) {
                    // std::cout << "D";std::cout.flush();

                    Mat12x12d elmH;
                    Vec12d elmD;
                    FEM_Scaler elmObj;

                    _intPtr->EvalElmObjDerivJacobi(attrbs,
                        muscle->_forceModel,
                        damp->_dampForce,
                        elm_traj,
                        &elmObj,elmD,elmH,false);

                    auto elmTrajTmp = elm_traj;
                    auto& curFrame = elmTrajTmp[clen - 1];

                    Vec12d elmDTmp;
                    FEM_Scaler elmObjTmp;
                    Mat12x12d elmHfd;
                    Vec12d elmDfd;

                    for(size_t i = 0;i < 12;++i){
                        auto elmTrajTmp = elm_traj;
                        auto& curFrame = elmTrajTmp[clen - 1];
                        FEM_Scaler step = curFrame[i] * 1e-10;
                        step = fabs(step) < 1e-10 ? 1e-10 : step;
                        curFrame[i] += step;

                        _intPtr->EvalElmObjDeriv(attrbs,
                            muscle->_forceModel,
                            damp->_dampForce,
                            elmTrajTmp,
                            &elmObjTmp,elmDTmp);

                        elmHfd.col(i) = (elmDTmp - elmD) / step;
                        elmDfd[i] = (elmObjTmp - elmObj) / step;
                    }

                    FEM_Scaler H_error = (elmHfd - elmH).norm() / elmH.norm();
                    FEM_Scaler D_error = (elmD - elmDfd).norm() / elmDfd.norm();

                    if(H_error > 1e-5){
                        std::cout << "ELM_ID : " << elm_id << std::endl;
                        std::cout << "H_ERROR : " << H_error << std::endl;
                        std::cout << "D_ERROR : " << D_error << std::endl;

                        // std::cout << "elmHfd : " << std::endl << elmHfd << std::endl;
                        // std::cout << "elmH : " << std::endl << elmH << std::endl;

                        // std::cout << "elmDInv : " << std::endl << mesh->_elmMinv[elm_id] << std::end;
                        // std::cout << "elmMInv : " << std::endl << mesh->_elmMinv[elm_id] << std::endl;

                        _intPtr->EvalElmObjDerivJacobi(attrbs,
                            muscle->_forceModel,
                            damp->_dampForce,
                            elm_traj,
                            &_objBuffer[elm_id],_derivBuffer[elm_id],_HBuffer[elm_id],false,true);

                        // std::cout << "H_diff : " << std::endl << elmHfd - elmH << std::endl;

                        // std::cout << "elmD : " << elmD.transpose() << std::endl;
                        // std::cout << "elmDfd:" << elmDfd.transpose() << std::endl;

                        throw std::runtime_error("INT_ERROR");
                    }
                }    
            }

            // throw std::runtime_error("CHECK");

            deriv.setZero();
            HValBuffer.setZero();
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                auto tet = mesh->_mesh->quads[elm_id];
                obj += _objBuffer[elm_id];
                // std::cout << "ELM_D_H : " << elm_id << "\t" << _derivBuffer[elm_id].norm() << "\t" << _HBuffer[elm_id].norm() << std::endl;
                AssembleElmVector(tet,_derivBuffer[elm_id],deriv);
                AssembleElmMatrixAdd(tet,_HBuffer[elm_id],mesh->MapHMatrixRef(HValBuffer.data()));

            }

            // std::cout << "OUT_D : " << deriv.norm() << std::endl;
            // std::cout << "OUT_H : " << HValBuffer.norm() << std::endl;
            // throw std::runtime_error("CHECK");
            return obj;
    }

    void RetrieveElmVector(const zeno::vec4i& elm,Vec12d& elm_vec,const VecXd& global_vec) const{  
        for(size_t i = 0;i < 4;++i)
            elm_vec.segment(i*3,3) = global_vec.segment(elm[i]*3,3);
    } 

    void AssembleElmVector(const zeno::vec4i& elm,const Vec12d& elm_vec,VecXd& global_vec) const{
        for(size_t i = 0;i < 4;++i)
            global_vec.segment(elm[i]*3,3) += elm_vec.segment(i*3,3);
    }

    void AssembleElmMatrixAdd(const zeno::vec4i& elm,const Mat12x12d& elm_H,Eigen::Map<SpMat> H) const{
        for(size_t i = 0;i < 4;++i) {
            for(size_t j = 0;j < 4;++j)
                for (size_t r = 0; r < 3; ++r)
                    for (size_t c = 0; c < 3; ++c)
                        H.coeffRef(elm[i] * 3 + r, elm[j] * 3 + c) += elm_H(i * 3 + r, j * 3 + c);
        } 
    }

};
struct GetCurrentFrame : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<PrimitiveObject>("femGeo");
        auto integrator = get_input<FEMIntegrator>("intIn");
        const auto& frame = integrator->GetCurrentFrame();

        auto resGeo = std::make_shared<PrimitiveObject>();
        auto &pos = resGeo->add_attr<zeno::vec3f>("pos");
        auto &btags = resGeo->add_attr<float>("btag");

        auto& fiberDirs = resGeo->add_attr<zeno::vec3f>("fiberDir");
        const auto& refFiberDirs = mesh->attr<zeno::vec3f>("fiberDir");
   
        for(size_t i = 0;i < mesh->size();++i){
            auto vert = frame.segment(i*3,3);
            pos.emplace_back(vert[0],vert[1],vert[2]);
            btags.emplace_back(mesh->attr<float>("btag")[i]);
        }
        for(int i=0;i < mesh->quads.size();++i){
            auto tet = mesh->quads[i];
            resGeo->tris.emplace_back(tet[0],tet[1],tet[2]);
            resGeo->tris.emplace_back(tet[1],tet[3],tet[2]);
            resGeo->tris.emplace_back(tet[0],tet[2],tet[3]);
            resGeo->tris.emplace_back(tet[0],tet[3],tet[1]);
        }

        fiberDirs.resize(mesh->size(),zeno::vec3f(0));
        for(size_t i = 0;i < mesh->quads.size();++i){
            const auto& tet = mesh->quads[i];
            // compute the deformation gradient
            Mat4x4d G,M;
            for(size_t i = 0;i < 4;++i){
                G.col(i) << mesh->verts[tet[i]][0],mesh->verts[tet[i]][1],mesh->verts[tet[i]][2],1.0;
                M.col(i) << frame.segment(tet[i]*3,3),1.0;
            }
            G = M * G.inverse();
            auto F = G.topLeftCorner(3,3);  

            FEM_Scaler vol = fabs(G.determinant()) / 6;

            for(size_t i = 0;i < 4;++i){
                Vec3d deformFiber;
                deformFiber << refFiberDirs[tet[i]][0],refFiberDirs[tet[i]][1],refFiberDirs[tet[i]][2];
                deformFiber = F * deformFiber;
                fiberDirs[tet[i]] += vol * zeno::vec3f(deformFiber[0],deformFiber[1],deformFiber[2]);
            }
        }

        for(size_t i = 0;i < mesh->size();++i)
            fiberDirs[i] /= zeno::length(fiberDirs[i]);

        set_output("frame",std::move(resGeo));
    }
};


ZENDEFNODE(GetCurrentFrame,{
    {"femGeo","intIn"},
    {"frame"},
    {},
    {"FEM"},
});

struct MakeFEMIntegrator : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<PrimitiveObject>("femGeo");
        auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto inttype = std::get<std::string>(get_param("integType"));

        auto res = std::make_shared<FEMIntegrator>();
        if(inttype == "BackwardEuler")
            res->_intPtr = std::make_shared<BackEulerIntegrator>();
        else if(inttype == "QuasiStatic")
            res->_intPtr = std::make_shared<QuasiStaticSolver>();

        res->_intPtr->SetGravity(Vec3d(gravity[0],gravity[1],gravity[2]));
        res->_intPtr->SetTimeStep(dt);
        res->_traj.resize(res->_intPtr->GetCouplingLength(),
                    VecXd::Zero(mesh->size() * 3));
        for(size_t i = 0;i < res->_intPtr->GetCouplingLength();++i)
            for(size_t j = 0;j < mesh->size();++j){
                auto& vert = mesh->verts[j];
                res->_traj[i][j*3 + 0] = vert[0];
                res->_traj[i][j*3 + 1] = vert[1];
                res->_traj[i][j*3 + 2] = vert[2];
            }

        res->_stepID = 0;

        size_t nm_elms = mesh->quads.size();

        res->_extForce.resize(mesh->size() * 3);res->_extForce.setConstant(0);
        res->_objBuffer.resize(nm_elms);
        res->_derivBuffer.resize(nm_elms);

        // by default, no plastic behavior
        res->kinematic_hardening_shifts.resize(nm_elms,Vec3d::Zero());
        res->plastic_strains.resize(nm_elms,Vec3d::Zero());
        res->PSs.resize(nm_elms,Mat3x3d::Zero());
        res->init_stresses.resize(nm_elms,Vec3d::Zero());
        res->init_strains.resize(nm_elms,Vec3d::Zero());
        res->kinimatic_hardening_coeffs.resize(nm_elms,0);
        res->yielding_stresses.resize(nm_elms,-1);
        res->restoring_strains.resize(nm_elms,1e6);// default setting, no restoring strain
        res->fail_strains.resize(nm_elms,1e6); // default setting, no fail strain
        res->failed.resize(nm_elms,false);
        res->the_strains_failed.resize(nm_elms,Vec3d::Zero()); // only effective when the material break
        res->Fs_failed.resize(nm_elms,Mat3x3d::Zero());
        res->rms.resize(nm_elms,0);
        res->vms.resize(nm_elms,0);

        res->uniformAct = Vec3d::Ones();

        FEM_Scaler ts = 0;
        for(size_t i = 0;i < mesh->size();++i){
            float vcl = mesh->attr<float>("cm")[i];
            ts += vcl*vcl;
        }

        ts = sqrt(ts);
        std::cout << "CL : " << "\t" << ts << std::endl;

        set_output("FEMIntegrator",res);
    }
};

ZENDEFNODE(MakeFEMIntegrator,{
    {{"femGeo"},{"gravity"},{"dt"}},
    {"FEMIntegrator"},
    {{"enum BackwardEuler QuasiStatic", "integType", "BackwardEuler"}},
    {"FEM"},
});

struct SetUniformActivation : zeno::INode {
    virtual void apply() override {
        auto intIn = get_input<FEMIntegrator>("intIn");
        auto act = get_input<zeno::NumericObject>("uniformAct")->get<zeno::vec3f>();

        intIn->uniformAct << act[0],act[1],act[2];

        set_output("intOut",intIn);
    }
};

ZENDEFNODE(SetUniformActivation,{
    {{"intIn"},{"uniformAct"}},
    {"intOut"},
    {},
    {"FEM"},
});

struct SetUniformPlasticParam : zeno::INode {
    virtual void apply() override {
        auto intIn = get_input<FEMIntegrator>("intIn");
        auto khs = get_input<zeno::NumericObject>("khs")->get<zeno::vec3f>();
        auto ps = get_input<zeno::NumericObject>("ps")->get<zeno::vec3f>();
        auto init_stress = get_input<zeno::NumericObject>("initStress")->get<zeno::vec3f>();
        auto init_strain = get_input<zeno::NumericObject>("initStrain")->get<zeno::vec3f>();

        auto khc = get_input<zeno::NumericObject>("khc")->get<float>();
        auto ys = get_input<zeno::NumericObject>("ys")->get<float>();
        auto rs = get_input<zeno::NumericObject>("rs")->get<float>();
        auto fs = get_input<zeno::NumericObject>("fs")->get<float>();

        size_t nm_elms = intIn->kinematic_hardening_shifts.size();

        for(size_t i = 0;i < nm_elms;++i){
            intIn->kinematic_hardening_shifts[i] << khs[0],khs[1],khs[2];
            intIn->plastic_strains[i] << ps[0],ps[1],ps[2];
            intIn->init_stresses[i] << init_stress[0],init_stress[1],init_stress[2];
            intIn->init_strains[i] << init_strain[0],init_strain[1],init_strain[2];
            intIn->kinimatic_hardening_coeffs[i] = khc;

            intIn->yielding_stresses[i] = ys;
            intIn->restoring_strains[i] = rs;
            intIn->fail_strains[i] = fs;
        }

        // std::cout << "SET : " << intIn->init_strains[0].transpose() << std::endl;
        // throw std::runtime_error("CHECK");

        set_output("intOut",std::move(intIn));
    }
};

ZENDEFNODE(SetUniformPlasticParam,{
    {{"intIn"},{"khs"},{"ps"},{"initStress"},{"initStrain"},{"khc"},{"ys"},{"rs"},{"fs"}},
    {"intOut"},
    {},
    {"FEM"},
});

struct SetExternalForce : zeno::INode {
    virtual void apply() override {
        auto intPtr = get_input<FEMIntegrator>("intIn");
        auto nodalForce = get_input<zeno::NumericObject>("nodalForce")->get<zeno::vec3f>();
        auto VertIDs = get_input<zeno::ListObject>("VertIds")->get<std::shared_ptr<NumericObject>>();

        for(size_t i = 0;i < VertIDs.size();++i){
            size_t vertID = VertIDs[i]->get<int>();
            intPtr->_extForce.segment(vertID*3,3) << nodalForce[0],nodalForce[1],nodalForce[2];
        }     

        set_output("intOut",intPtr);   
    }
};

ZENDEFNODE(SetExternalForce,{
    {{"intIn"},{"nodalForce"},{"VertIds"}},
    {"intOut"},
    {},
    {"FEM"},
});

struct ApplyTranformOnPoint : zeno::INode {
    virtual void apply() override {
        auto trans = get_input<TransformMatrix>("T");
        auto point = get_input<zeno::NumericObject>("P")->get<zeno::vec3f>();

        Vec4d pv;
        pv << point[0],point[1],point[2],1.0;
        pv = trans->Mat * pv;

        auto res = std::make_shared<zeno::NumericObject>();
        res->set<zeno::vec3f>(zeno::vec3f(pv[0],pv[1],pv[2]));

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(ApplyTranformOnPoint,{
    {{"T"},{"P"}},
    {"res"},
    {},
    {"FEM"},
});

struct SetBoundaryMotion : zeno::INode {
       virtual void apply() override {
        auto intPtr = get_input<FEMIntegrator>("intIn");
        auto mesh = get_input<PrimitiveObject>("refGeo");
        auto CT = get_input<TransformMatrix>("CT");
        auto FT = get_input<TransformMatrix>("FT");

        const auto& btags = mesh->attr<float>("btag");
        for(size_t i = 0;i < mesh->size();++i){
            const auto& ref_vert = mesh->attr<zeno::vec3f>("pos")[i];
            Vec4d vert_w;vert_w << ref_vert[0],ref_vert[1],ref_vert[2],1.0;
            switch((int)btags[i]){
            case 0 : break;
            case 1 : 
                vert_w = CT->Mat * vert_w;
                intPtr->GetCurrentFrame().segment(i*3,3) = vert_w.segment(0,3);
                break;
            case 2 :
                vert_w = FT->Mat * vert_w;
                intPtr->GetCurrentFrame().segment(i*3,3) = vert_w.segment(0,3);
                break;
            default:
                std::cerr << "INVALID BTAG : " << btags[i] << std::endl;
                throw std::runtime_error("INVALID BTAGS");
            }
        }
        set_output("intOut",intPtr);   
    } 
};

ZENDEFNODE(SetBoundaryMotion,{
    {{"intIn"},{"refGeo"},{"CT"},{"FT"}},
    {"intOut"},
    {},
    {"FEM"},
});


struct TranslateBoundary : zeno::INode {
       virtual void apply() override {
        auto intPtr = get_input<FEMIntegrator>("intIn");
        auto mesh = get_input<PrimitiveObject>("refGeo");
        auto CT = get_input<zeno::NumericObject>("CT")->get<zeno::vec3f>();
        auto FT = get_input<zeno::NumericObject>("FT")->get<zeno::vec3f>();

        const auto& btags = mesh->attr<float>("btag");
        for(size_t i = 0;i < mesh->size();++i){
            const auto& ref_vert = mesh->attr<zeno::vec3f>("pos")[i];
            Vec3d CTv,FTv;
            switch((int)btags[i]){
            case 0 : break;
            case 1 : 
                CTv << CT[0],CT[1],CT[2];
                intPtr->GetCurrentFrame().segment(i*3,3) += CTv;
                break;
            case 2 :
                FTv << FT[0],FT[1],FT[2];
                intPtr->GetCurrentFrame().segment(i*3,3) += FTv;
                break;
            default:
                std::cerr << "INVALID BTAG : " << btags[i] << std::endl;
                throw std::runtime_error("INVALID BTAGS");
            }
        }
        set_output("intOut",intPtr);   
    } 
};


ZENDEFNODE(TranslateBoundary,{
    {{"intIn"},{"refGeo"},{"CT"},{"FT"}},
    {"intOut"},
    {},
    {"FEM"},
});

struct DoStep : zeno::INode {
       virtual void apply() {
        auto intPtr = get_input<FEMIntegrator>("intIn");
        intPtr->_stepID++;
        set_output("intOut",intPtr);   
    }     
};

ZENDEFNODE(DoStep,{
    {{"intIn"}},
    {"intOut"},
    {},
    {"FEM"},
});

struct RetrieveRigidTransform : zeno::INode {
    virtual void apply() override {
        auto objRef = get_input<zeno::PrimitiveObject>("refObj");
        auto objNew = get_input<zeno::PrimitiveObject>("newObj");

        Mat4x4d refTet,newTet;
        for(size_t i = 0;i < 4;++i){
            refTet.col(i) << objRef->verts[i][0],objRef->verts[i][1],objRef->verts[i][2],1.0;
            newTet.col(i) << objNew->verts[i][0],objNew->verts[i][1],objNew->verts[i][2],1.0;
        }

        Mat4x4d T = newTet * refTet.inverse();

        auto ret = std::make_shared<TransformMatrix>();
        ret->Mat = T;

        set_output("T",std::move(ret));
    }
};

ZENDEFNODE(RetrieveRigidTransform,{
    {{"refObj"},{"newObj"}},
    {"T"},
    {},
    {"FEM"},
});

struct BindingIndices : zeno::IObject {
    BindingIndices() = default;
    std::vector<int> _closedBindPoints;
    std::vector<int> _farBindPoints;
};


struct OutputVec2List : zeno::INode {
    virtual void apply() override {
        auto vec2List = get_input<zeno::ListObject>("vec2list");
        auto outfile = get_input<zeno::StringObject>("outfile")->get();
        size_t nm = vec2List->get<std::shared_ptr<zeno::NumericObject>>().size();
        
        std::ofstream fout(outfile);
        if(!fout.is_open()){
            std::cerr << "FAIL OPENING : " << outfile << std::endl;
            throw std::runtime_error("FAIL OPENING TRAJ FILE");
        }

        try{
            for(size_t i = 0;i < nm;++i){
                const auto v2 = vec2List->get<std::shared_ptr<zeno::NumericObject>>()[i]->get<zeno::vec2f>();
                fout << v2[0] << "\t" << v2[1] << std::endl;
            }

            fout.close();
        }catch(const std::exception& e){
            std::cerr << e.what() << std::endl;
        }
    }
};
ZENDEFNODE(OutputVec2List,{
    {{"vec2list"},{"readpath","outfile"}},
    {},
    {},
    {"FEM"},
});
struct MakeBindingIndices : zeno::INode{
    virtual void apply() override {
        auto closedBindIndices = get_input<zeno::ListObject>("cbIndices");
        auto farBindIndices = get_input<zeno::ListObject>("fbIndices");

        auto res = std::make_shared<BindingIndices>();

        size_t nmcbps = closedBindIndices->get().size();
        size_t nmfbps = farBindIndices->get().size();

        res->_closedBindPoints.resize(nmcbps);
        res->_farBindPoints.resize(nmfbps);

        for(size_t i = 0;i < nmcbps;++i){
            res->_closedBindPoints[i] = closedBindIndices->get<std::shared_ptr<zeno::NumericObject>>()[i]->get<int>();
            //res->_closedBindPoints[i] = closedBindIndices->arr[i]->get<zeno::NumericObject>().get<int>();
        }
        for(size_t i = 0;i < nmfbps;++i){
            res->_farBindPoints[i] = farBindIndices->get<std::shared_ptr<zeno::NumericObject>>()[i]->get<int>();
            //res->_farBindPoints[i] = farBindIndices->arr[i]->get<zeno::NumericObject>().get<int>();
        }

        // std::cout << "OUTPUT BINDING INDICES : " << std::endl;
        // for(size_t i = 0;i < res->_closedBindPoints.size();++i){
        //     std::cout << "C<" << i <<  "> :\t" << res->_closedBindPoints[i] << std::endl;
        // }

        // for(size_t i = 0;i < res->_farBindPoints.size();++i){
        //     std::cout << "F<" << i << "> :\t" << res->_farBindPoints[i] << std::endl;
        // }

        set_output("BindIndices",std::move(res));
    }
};
ZENDEFNODE(MakeBindingIndices,{
    {{"cbIndices"},{"fbIndices"}},
    {"BindIndices"},
    {},
    {"FEM"},
});

struct LoadBindingIndicesFromFile : zeno::INode {
    virtual void apply() override {
        auto bindFile = get_input<zeno::StringObject>("BindFile")->get();

        auto res = std::make_shared<BindingIndices>();

        size_t nmClosedBind,nmFarBind;
        std::ifstream bind_fin;
        try{
            bind_fin.open(bindFile.c_str());
            if (!bind_fin.is_open()) {
                std::cerr << "ERROR::NODE::FAILED::" << bindFile << std::endl;
            }
            bind_fin >> nmClosedBind >> nmFarBind;
            res->_closedBindPoints.resize(nmClosedBind);
            res->_farBindPoints.resize(nmFarBind);

            for(size_t i = 0;i < nmClosedBind;++i)
                bind_fin >> res->_closedBindPoints[i];

            for(size_t i = 0;i < nmFarBind;++i)
                bind_fin >> res->_farBindPoints[i];
            bind_fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }     

        set_output("res",std::move(res));   
    }
};

ZENDEFNODE(LoadBindingIndicesFromFile,{
    {{"readpath","BindFile"}},
    {"res"},
    {},
    {"FEM"},
});



// struct SetFEMBindingPoints : zeno::INode {
//     virtual void apply() override {
//         auto bindPoints = get_input<BindingIndices>("bindPoints");
//         auto mesh = get_input<FEMMesh>("mesh");

//         mesh->_closeBindPoints = bindPoints->_closedBindPoints;
//         mesh->_farBindPoints = bindPoints->_farBindPoints;


//         auto& btag = mesh->_mesh->attr<float>("btag");
//         btag.resize(mesh->_mesh->size(),0.0);
//         for(size_t i = 0;i < mesh->_closeBindPoints.size();++i)
//             btag[mesh->_closeBindPoints[i]] = 1.0;
//         for(size_t i = 0;i < mesh->_farBindPoints.size();++i)
//             btag[mesh->_farBindPoints[i]] = 2.0;            

//         size_t nm_con_vertices = mesh->_closeBindPoints.size() + mesh->_farBindPoints.size();
//         // std::cout << "nm_con_vertices : " << nm_con_vertices << std::endl;
//         // for(size_t i = 0;i < mesh->_closeBindPoints.size();++i)
//         //     std::cout << "C<" << i << "> : " << mesh->_closeBindPoints[i] << std::endl;
//         // for(size_t i = 0;i < mesh->_farBindPoints.size();++i)
//         //     std::cout << "F<" << i << "> : " << mesh->_farBindPoints[i] << std::endl;


//         mesh->_bouDoFs.clear();
//         for(size_t i = 0;i < mesh->_closeBindPoints.size();++i){
//             size_t vert_idx = mesh->_closeBindPoints[i];
//             mesh->_bouDoFs.emplace_back(vert_idx * 3 + 0);
//             mesh->_bouDoFs.emplace_back(vert_idx * 3 + 1);
//             mesh->_bouDoFs.emplace_back(vert_idx * 3 + 2);
//         }
//         for(size_t i = 0;i < mesh->_farBindPoints.size();++i){
//             size_t vert_idx = mesh->_farBindPoints[i];
//             mesh->_bouDoFs.emplace_back(vert_idx * 3 + 0);
//             mesh->_bouDoFs.emplace_back(vert_idx * 3 + 1);
//             mesh->_bouDoFs.emplace_back(vert_idx * 3 + 2);
//         }

//         std::sort(mesh->_bouDoFs.begin(),mesh->_bouDoFs.end(),std::less<int>());

//         std::cout << "UPDATE DOFS MAPPING" << std::endl;
//         mesh->UpdateDoFsMapping();
//         std::cout << "FINISH UPDATING DOFS MAPPING" << std::endl;

//         set_output("meshOut",std::move(mesh));
//     }
// };

// ZENDEFNODE(SetFEMBindingPoints, {
//     {"bindPoints","mesh"},
//     {"meshOut"},
//     {},
//     {"FEM"},
// });

struct StrainStraj : zeno::IObject {
    StrainStraj() = default;
    std::vector<FEM_Scaler> straj;
};

struct ReadStrainTrajFromFile : zeno::INode {
    virtual void apply() override {
        auto filename = get_input<zeno::StringObject>("trajfile")->get();
        auto res = std::make_shared<StrainStraj>();

        std::ifstream fin(filename);
        if(!fin.is_open()){
            std::cerr << "FAILED OPENING TRAJ : " << filename << std::endl;
            throw std::runtime_error("FAILED OPENING FILE");
        }

        try{
            res->straj.clear();
            while(true){
                if(fin.eof())
                    break;
                FEM_Scaler strain,stress;
                fin >> strain >> stress;
                res->straj.push_back(strain);
            }

            fin.close();
        }catch(const std::exception& e){
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("READ FAIL");
        }

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(ReadStrainTrajFromFile, {
    {{"readpath","trajfile"}},
    {"res"},
    {},
    {"FEM"},
});

struct GetTrajStrain : zeno::INode {
    virtual void apply() override {
        auto trajs = get_input<StrainStraj>("straj");
        auto frame_id = get_input<zeno::NumericObject>("frameID")->get<int>();

        auto res = std::make_shared<zeno::NumericObject>();
        res->set<float>(trajs->straj[frame_id]);

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(GetTrajStrain, {
    {"straj","frameID"},
    {"res"},
    {},
    {"FEM"},
});



// the boundary motion and external force are set before the node is applied.
struct SolveEquaUsingNRSolver : zeno::INode {
    VecXd r,ruc;
    VecXd HBuffer,HucBuffer;
    VecXd dp,dpuc;
    bool analyized_pattern = false;
    Eigen::SparseLU<SpMat> _LUSolver;
    Eigen::SimplicialLDLT<SpMat> _LDLTSolver;

    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("mesh");
        auto force_model = get_input<MuscleModelObject>("muscleForce");
        auto damping_model = get_input<DampingForceModel>("dampForce");
        auto integrator = get_input<FEMIntegrator>("integrator");

        int max_iters = get_param<int>("maxNRIters");
        int max_linesearch = get_param<int>("maxBTLs");
        float c1 = get_param<float>("ArmijoCoeff");
        float c2 = get_param<float>("CurvatureCoeff");
        float beta = get_param<float>("BTL_shrinkingRate");
        float epsilon = get_param<float>("epsilon");

        std::vector<Vec2d> wolfeBuffer;
        wolfeBuffer.resize(max_linesearch);

        int search_idx = 0;

        r.resize(mesh->_mesh->size() * 3);
        ruc.resize(mesh->_freeDoFs.size());
        dp.resize(mesh->_mesh->size() * 3);
        dpuc.resize(mesh->_freeDoFs.size());
        HBuffer.resize(mesh->_connMatrix.nonZeros());
        HucBuffer.resize(mesh->_freeConnMatrix.nonZeros());

        size_t iter_idx = 0;

        FEM_Scaler ruc0 = 0;
        FEM_Scaler e_start = 0;

        FEM_Scaler stop_error = 0;

        do{
            // break;
            FEM_Scaler e0,e1,eg0;

            e0 = integrator->EvalObjDerivHessian(mesh,force_model,damping_model,r,HBuffer,true);
            FEM_Scaler stopError = 0;
            if(iter_idx == 0){
                std::vector<FEM_Scaler> vert_tol;
                std::vector<FEM_Scaler> vert_vol;
                vert_tol.resize(mesh->_mesh->size(),0);
                vert_vol.resize(mesh->_mesh->size(),0);
                for(size_t i = 0;i < mesh->_mesh->quads.size();++i){
                    Vec12d tetShape;
                    const auto& tet = mesh->_mesh->quads[i];
                    integrator->RetrieveElmVector(tet,tetShape,integrator->GetCurrentFrame());

                    Mat3x3d F;
                    BaseIntegrator::ComputeDeformationGradient(mesh->_elmMinv[i],tetShape,F);

                    TetAttributes attrs;
                    integrator->AssignElmAttribs(i,attrs,mesh,Vec12d::Zero());

                    FEM_Scaler psi;
                    Vec9d dpsi;
                    Mat9x9d ddpsi;
                    force_model->_forceModel->ComputePsiDerivHessian(attrs,F,psi,dpsi,ddpsi,false);

                    // std::cout << "ELM_H<" << i << "> :\t" << ddpsi.norm() << "\t" << mesh->_elmVolume[i] << std::endl;

                    vert_tol[tet[0]] += mesh->_elmVolume[i]/4 * ddpsi.norm();
                    vert_tol[tet[1]] += mesh->_elmVolume[i]/4 * ddpsi.norm();
                    vert_tol[tet[2]] += mesh->_elmVolume[i]/4 * ddpsi.norm();
                    vert_tol[tet[3]] += mesh->_elmVolume[i]/4 * ddpsi.norm();

                    vert_vol[tet[0]] += mesh->_elmVolume[i]/4;
                    vert_vol[tet[1]] += mesh->_elmVolume[i]/4;
                    vert_vol[tet[2]] += mesh->_elmVolume[i]/4;
                    vert_vol[tet[3]] += mesh->_elmVolume[i]/4;
                }

                for(size_t i = 0;i < mesh->_mesh->size();++i){
                    vert_tol[i] /= vert_vol[i];
                    // std::cout << "vert_tol<" << i << "> : \t" << vert_tol[i] << "\t";
                    vert_tol[i] *= mesh->_mesh->attr<float>("cm")[i];
                    // std::cout << mesh->_mesh->attr<float>("cm")[i] << std::endl;
                    stop_error += vert_tol[i];
                }

                stop_error *= epsilon;
                stop_error *= sqrt(mesh->_mesh->size());

                std::cout << "STOP_ERROR : " << stop_error << std::endl;                
            }

            bool debug_global = false;
            if(debug_global){
                std::cout << "D : " << std::endl;
                FEM_Scaler e_cmp;
                VecXd r_cmp = r;
                VecXd HBuffer_cmp = HBuffer;
                e_cmp = integrator->EvalObjDerivHessian(mesh,force_model,damping_model,r_cmp,HBuffer_cmp,false);


                VecXd ruc_cmp = ruc;
                VecXd HucBuffer_cmp = HucBuffer;
                MatHelper::RetrieveDoFs(r_cmp.data(),ruc_cmp.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());
                MatHelper::RetrieveDoFs(HBuffer_cmp.data(),HucBuffer_cmp.data(),mesh->_SpMatFreeDoFs.size(),mesh->_SpMatFreeDoFs.data());

                FEM_Scaler e_tmp;
                VecXd r_tmp = r;

                VecXd curFrameCopy = integrator->GetCurrentFrame();

                MatXd HucBuffer_fd = MatXd(ruc.size(),ruc.size());
                VecXd r_fd = r;
                VecXd ruc_fd = ruc;
                VecXd ruc_tmp = ruc;

                for(size_t i = 0;i < mesh->_freeDoFs.size();++i){
                    size_t idx = mesh->_freeDoFs[i];
                    integrator->GetCurrentFrame() = curFrameCopy;
                    FEM_Scaler step = integrator->GetCurrentFrame()[idx] * 1e-8;
                    step = fabs(step) < 1e-8 ? 1e-8 : step;
                    integrator->GetCurrentFrame()[idx] += step;
                    e_tmp = integrator->EvalObjDeriv(mesh,force_model,damping_model,r_tmp);

                    ruc_fd[i] = (e_tmp - e_cmp) / step;
                    MatHelper::RetrieveDoFs(r_tmp.data(),ruc_tmp.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());
                    HucBuffer_fd.col(i) = (ruc_tmp - ruc_cmp)/step;
                }

                FEM_Scaler H_error = (HucBuffer_fd - mesh->MapHucMatrix(HucBuffer_cmp.data()).toDense()).norm() / HucBuffer_cmp.norm();
                FEM_Scaler D_error = (ruc_fd - ruc_cmp).norm();

                // if(H_error > 1e-3){
                    std::cout << "H_ERROR_GLOBAL : " << H_error << std::endl;
                    std::cout << "R_ERROR_GLOBAL : " << D_error << "\t" << ruc_fd.norm() << "\t" << ruc_cmp.norm() << std::endl;

                //     throw std::runtime_error("H_ERROR");
                // }
            }


            if(std::isnan(r.norm()) || std::isnan(HBuffer.norm())){
                std::cerr << "NAN VALUE DETECTED : " << r.norm() << "\t" << HBuffer.norm() << std::endl;
                throw std::runtime_error("NAN VALUE DETECTED");
            }

            MatHelper::RetrieveDoFs(r.data(),ruc.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());
            MatHelper::RetrieveDoFs(HBuffer.data(),HucBuffer.data(),mesh->_SpMatFreeDoFs.size(),mesh->_SpMatFreeDoFs.data());


            if(iter_idx == 0){
                ruc0 = ruc.norm();
                e_start = e0;
            }

            if(
                // ruc.norm() < epsilon
                ruc.norm() < stop_error
             //|| ruc.norm() < ruc0 * 5e-4
             ){
                // std::cout << "BREAK WITH RUC = " << ruc.norm()  << "\t" << ruc0 << std::endl;
                break;
            }
            ruc *= -1;

            clock_t begin_solve = clock();

            std::cout << "BEGIN_SOVLE" << std::endl;

            if(!analyized_pattern){
                _LDLTSolver.analyzePattern(mesh->MapHucMatrix(HucBuffer.data()));
                analyized_pattern = true;
            }

            _LDLTSolver.factorize(mesh->MapHucMatrix(HucBuffer.data()));
            dpuc = _LDLTSolver.solve(ruc);


            clock_t end_solve = clock();



            eg0 = -dpuc.dot(ruc);

            // if(fabs(eg0) < stop_error * stop_error){
            //     // std::cout << "BREAK WITH EG0 = " << eg0 << "\t" << ruc.norm() << std::endl;
            //     break;
            // }

            if(eg0 > 0){
                std::cout << "eg0 = " << eg0 << std::endl;
                throw std::runtime_error("non-negative descent direction");
            }
            dp.setZero();
            MatHelper::UpdateDoFs(dpuc.data(),dp.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());
            bool do_line_search = true;
            size_t search_idx = 0;
            if(!do_line_search)
                integrator->GetCurrentFrame() += dp;
            else{
                search_idx = 0;

                FEM_Scaler alpha = 2.0f;
                FEM_Scaler beta = 0.5f;
                FEM_Scaler c1 = 0.001f;

                double armijo_condition;
                do{
                    if(search_idx != 0)
                        integrator->GetCurrentFrame() -= alpha * dp;
                    alpha *= beta;
                    integrator->GetCurrentFrame() += alpha * dp;
                    e1 = integrator->EvalObj(mesh,force_model,damping_model);
                    ++search_idx;
                    wolfeBuffer[search_idx-1](0) = (e1 - e0)/alpha;
                    wolfeBuffer[search_idx-1](1) = eg0;

                    armijo_condition = double(e1) - double(e0) - double(c1)*double(alpha)*double(eg0);
                }while(/*(e1 > e0 + c1*alpha*eg0)*/ armijo_condition > 0.0f /* || (fabs(eg1) > c2*fabs(eg0))*/ && (search_idx < max_linesearch));

                if(search_idx == max_linesearch){
                    std::cout << "LINESEARCH EXCEED" << std::endl;
                    for(size_t i = 0;i < max_linesearch;++i)
                        std::cout << "idx:" << i << "\t" << wolfeBuffer[i].transpose() << std::endl;
                    break;
                }
            }
            std::cout << "SOLVE TIME : " << (float)(end_solve - begin_solve)/CLOCKS_PER_SEC << "\t" << ruc0 << "\t" << ruc.norm() << "\t" << eg0 << "\t" << search_idx << "\t" << e_start << "\t" << e0 << "\t" << e1 << std::endl;

            ++iter_idx;
        }while(iter_idx < max_iters);

        if(iter_idx == max_iters){
            std::cout << "MAX NEWTON ITERS EXCEED" << std::endl;
        }

        // std::cout << "FINISH STEPPING WITH RUC = " << ruc.norm() << "\t" << "NM_ITERS = " << iter_idx << std::endl;

        // OUTPUT THE STRESS FIELD
        auto stress_field = std::make_shared<zeno::PrimitiveObject>();
        auto& ppos = stress_field->add_attr<zeno::vec3f>("pos");
        auto& pstress = stress_field->add_attr<zeno::vec3f>("pstress");
        stress_field->resize(mesh->_mesh->quads.size());
        for(size_t i = 0;i < mesh->_mesh->quads.size();++i){
            // std::cout << "HERE1" << std::endl;
            const auto& tet = mesh->_mesh->quads[i];
            ppos[i] = zeno::vec3f(0.0);
            for(size_t j = 0;j < 4;++j)
                ppos[i] += mesh->_mesh->verts[tet[j]];
            ppos[i] /= 4;

            pstress[i] = zeno::vec3f(0.0);

            TetAttributes attrbs;
            integrator->AssignElmAttribs(i,attrbs,mesh,Vec12d::Zero());

            Vec12d u;
            integrator->RetrieveElmVector(tet,u,integrator->GetCurrentFrame());
            Mat3x3d F;
            BaseIntegrator::ComputeDeformationGradient(attrbs._Minv,u,F);

            FEM_Scaler psi;
            Vec9d dpsi;

            force_model->_forceModel->ComputePsiDeriv(attrbs,F,psi,dpsi);


            Mat3x3d respS = MatHelper::MAT(dpsi);
            Vec3d traction = respS * Vec3d(0.0,1.0,0.0);

            pstress[i] = zeno::vec3f(traction[0],traction[1],traction[2]);
            // std::cout << "HERE4" << std::endl;
        }

        set_output("stress_field",std::move(stress_field));

        set_output("intOut",std::move(integrator)); 

        std::cout << "FINISH STEPPING " << "\t" << iter_idx << std::endl;
    }
};

ZENDEFNODE(SolveEquaUsingNRSolver,{
    {"mesh","muscleForce","dampForce","integrator"},
    {"intOut","stress_field"},
    {{"int","maxNRIters","10"},{"int","maxBTLs","10"},{"float","ArmijoCoeff","0.01"},
        {"float","CurvatureCoeff","0.9"},{"float","BTL_shrinkingRate","0.5"},
        {"float","epsilon","1e-6"}
    },
    {"FEM"},
});


struct GetAverageAttrByTag : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto attrName = get_param<std::string>("attrname");
        auto tagName = get_param<std::string>("tagName");
        const auto& tags = prim->attr<float>(tagName);

        auto res_ptr = std::make_shared<zeno::NumericObject>();
        auto res = zeno::vec3f(0.0);

        size_t nm_elms = prim->size();
        size_t nm_tag = 0;
        for(size_t i = 0;i < nm_elms;++i){
            if(tags[i] == 1.0){
                res += prim->attr<zeno::vec3f>(attrName)[i];
                nm_tag++;
            }
        }
        res /= nm_tag;
        res_ptr->set<zeno::vec3f>(res);

        set_output("res",std::move(res_ptr));
    }
};
ZENDEFNODE(GetAverageAttrByTag,{
    {"prim"},
    {"res"},
    {{"string","attrname","attrName"},{"string","tagName","vtag"}
    },
    {"FEM"},
});

struct UpdatePlasticParam : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("mesh");
        auto force_model = get_input<MuscleModelObject>("muscleForce");
        auto integrator = get_input<FEMIntegrator>("integrator");

        if(dynamic_cast<PlasticForceModel*>(force_model->_forceModel.get())){
            size_t nm_elms = mesh->_mesh->quads.size();
            const auto& model = dynamic_cast<PlasticForceModel*>(force_model->_forceModel.get());

            clock_t begin_update = clock();

            FEM_Scaler avg_rm = 0;
            FEM_Scaler avg_vm = 0;

            // #pragma omp parallel for
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                TetAttributes attrs;
                attrs.pmp.init_stress = integrator->init_stresses[elm_id];
                attrs.pmp.init_strain = integrator->init_strains[elm_id];
                attrs.pmp.isotropic_hardening_coeff = 0;
                attrs.pmp.kinematic_hardening_coeff = integrator->kinimatic_hardening_coeffs[elm_id];
                attrs.pmp.kinematic_hardening_shift = integrator->kinematic_hardening_shifts[elm_id];
                attrs.pmp.plastic_strain = integrator->plastic_strains[elm_id];
                attrs.pmp.PS = integrator->PSs[elm_id];
                attrs.pmp.yield_stress = integrator->yielding_stresses[elm_id];
                attrs.pmp.restoring_strain = integrator->restoring_strains[elm_id]; 
                attrs.pmp.failed_strain = integrator->fail_strains[elm_id]; 
                attrs.pmp.failed = integrator->failed[elm_id];
                attrs.pmp.the_strain_failed = integrator->the_strains_failed[elm_id]; 
                attrs.pmp.F_failed = integrator->Fs_failed[elm_id];

                if(has_input("fiber")){
                    auto fiber = get_input<PrimitiveObject>("fiber");
                    const auto& orient = fiber->attr<zeno::vec3f>("orient")[elm_id];
                    attrs.emp.forient << orient[0],orient[1],orient[2];
                    zeno::vec3f act = fiber->attr<zeno::vec3f>("act")[elm_id];
                    Vec3d act_vec;act_vec << act[0],act[1],act[2];
                    Mat3x3d R = MatHelper::Orient2R(attrs.emp.forient);
                    attrs.emp.Act = R * act_vec.asDiagonal() * R.transpose();
                }else{
                    attrs.emp.forient << 1.0,0.0,0.0;
                    attrs.emp.Act =  Mat3x3d::Identity();
                }
                attrs.emp.E = mesh->_elmYoungModulus[elm_id];
                attrs.emp.nu = mesh->_elmPossonRatio[elm_id];

                const auto& tet = mesh->_mesh->quads[elm_id];
                Vec12d tet_shape = Vec12d::Zero();
                for(size_t j = 0;j < 4;++j){
                    size_t v_id = tet[j];
                    const auto& vert = integrator->GetCurrentFrame().segment(v_id*3,3);
                    tet_shape.segment(j*3,3) << vert[0],vert[1],vert[2];
                }

                Mat3x3d F;
                BaseIntegrator::ComputeDeformationGradient(mesh->_elmMinv[elm_id],tet_shape,F);

                FEM_Scaler rm;
                FEM_Scaler vm;
                model->UpdatePlasticParameters(elm_id,attrs,F,vm,rm);
                vm = 1+vm/attrs.pmp.yield_stress;

                integrator->kinematic_hardening_shifts[elm_id] = attrs.pmp.kinematic_hardening_shift;
                integrator->plastic_strains[elm_id] = attrs.pmp.plastic_strain;
                integrator->PSs[elm_id] = attrs.pmp.PS;
                integrator->init_stresses[elm_id] = attrs.pmp.init_stress;
                integrator->init_strains[elm_id] = attrs.pmp.init_strain;

                integrator->kinimatic_hardening_coeffs[elm_id] = attrs.pmp.kinematic_hardening_coeff;
                integrator->yielding_stresses[elm_id] = attrs.pmp.yield_stress;
                integrator->failed[elm_id] = attrs.pmp.failed;
                integrator->the_strains_failed[elm_id] = attrs.pmp.the_strain_failed;
                integrator->Fs_failed[elm_id] = attrs.pmp.F_failed;

                integrator->rms[elm_id] = rm;
                integrator->vms[elm_id] = vm;
            }  

            clock_t end_update = clock();
            // std::cout << "UPDATE_PLASTIC_TIME : " << (float)(end_update - begin_update) / CLOCKS_PER_SEC << std::endl;
        }

        set_output("intOut",std::move(integrator));
    }
};

ZENDEFNODE(UpdatePlasticParam,{
    {"mesh","muscleForce","integrator","fiber"},
    {"intOut"},
    {},
    {"FEM"},
});


struct GetPlasticStrain : zeno::INode {
    virtual void apply() override {
        auto deformation = get_input<PrimitiveObject>("deform");
        auto refmesh = get_input<FEMMesh>("refMesh");
        auto integrator = get_input<FEMIntegrator>("intIn");

        auto res = std::make_shared<PrimitiveObject>();
        auto& pos = res->add_attr<zeno::vec3f>("pos");
        auto& es = res->add_attr<float>("esigma");

        res->resize(refmesh->_mesh->quads.size());
        for(size_t i = 0;i < res->size();++i){
            const auto& tet = refmesh->_mesh->quads[i];
            pos[i] = zeno::vec3f(0.0);
            Vec12d tet_shape = Vec12d::Zero();
            for(size_t j = 0;j < 4;++j){
                size_t v_id = tet[j];
                const auto& vert = deformation->attr<zeno::vec3f>("pos")[v_id];
                pos[i] += vert;
                tet_shape.segment(j*3,3) << vert[0],vert[1],vert[2];
            }

            pos[i] /= 4;
            Vec3d s = integrator->plastic_strains[i];

            FEM_Scaler vm = compute_von_mises_strain(s);
            es[i] = vm;
        }

        set_output("res",std::move(res));
    }

    FEM_Scaler compute_von_mises_strain(const Vec3d& s) {
        FEM_Scaler s01 = s[0] - s[1];
        FEM_Scaler s02 = s[0] - s[2];
        FEM_Scaler s12 = s[1] - s[2];
        return sqrt(0.5 * (s01*s01 + s02*s02 + s12*s12));
    }
};

ZENDEFNODE(GetPlasticStrain,{
    {"deform","refMesh","intIn"},
    {"res"},
    {},
    {"FEM"},
});


struct GetPlasticState : zeno::INode {
    virtual void apply() override {
        auto intIn = get_input<FEMIntegrator>("intIn");
        auto refmesh = get_input<FEMMesh>("refMesh");

        auto res = std::make_shared<PrimitiveObject>();
        auto& pos = res->add_attr<zeno::vec3f>("pos");
        // auto& rpos = res->add_attr<zeno::vec3f>("rpos");
        auto& rms = res->add_attr<float>("rm");
        auto& vms = res->add_attr<float>("vm");

        auto& pstrain = res->add_attr<zeno::vec3f>("pstrain");
        auto& estrain = res->add_attr<zeno::vec3f>("estrain");
        auto& tstrain = res->add_attr<zeno::vec3f>("tstrain");

        res->resize(refmesh->_mesh->quads.size());
        for(size_t i = 0;i < res->size();++i){
            const auto& tet = refmesh->_mesh->quads[i];
            pos[i] = zeno::vec3f(0.0);
            // rpos[i] = zeno::vec3f(0.0);
            Vec12d tet_shape = Vec12d::Zero();
            for(size_t j = 0;j < 4;++j){
                size_t v_id = tet[j];
                const auto& vert = intIn->GetCurrentFrame().segment(v_id*3,3);
                pos[i] += zeno::vec3f(vert[0],vert[1],vert[2]);
                tet_shape.segment(j*3,3) = vert;

                // rpos[i] += refmesh->_mesh->verts[v_id];
            }

            // rpos[i] /= 4;
            pos[i] /= 4;

            // FEM_Scaler vm = compute_von_mises_strain(s);

            rms[i] = intIn->rms[i];
            vms[i] = intIn->vms[i];


            // if(i == 0){
            //     std::cout << "ELM_INI_STRAIN<" << i << "> : \t" << intIn->init_strains[i].transpose() << std::endl;
            //     std::cout << "ELM_<"  << i << "> : " << rms[i] << std::endl;
            // }


            Mat3x3d F;
            BaseIntegrator::ComputeDeformationGradient(refmesh->_elmMinv[i],tet_shape,F);
            Mat3x3d U,V;Vec3d s;
            DiffSVD::SVD_Decomposition(F,U,s,V);

            tstrain[i] = zeno::vec3f(s[0],s[1],s[2]);
            pstrain[i] = zeno::vec3f(intIn->plastic_strains[i][0],intIn->plastic_strains[i][1],intIn->plastic_strains[i][2]);

            // std::cout << "PS:" << pstrain[i][0] << "\t" << pstrain[i][1] << "\t" << pstrain[i][2] << std::endl;
            
            auto est = s - intIn->plastic_strains[i];
            // std::cout << "ES:" << est[0] << "\t" << est[1] << "\t" << est[2] << std::endl;
            estrain[i] = zeno::vec3f(est[0],est[1],est[2]);
        }

        set_output("res",std::move(res));        
    }
};

ZENDEFNODE(GetPlasticState,{
    {"refMesh","intIn"},
    {"res"},
    {},
    {"FEM"},
});

struct GetEffectiveStrain : zeno::INode {
    virtual void apply() override {
        auto deformation = get_input<PrimitiveObject>("deform");
        auto refmesh = get_input<FEMMesh>("refMesh");

        auto res = std::make_shared<PrimitiveObject>();
        auto& pos = res->add_attr<zeno::vec3f>("pos");
        auto& es = res->add_attr<float>("esigma");

        res->resize(refmesh->_mesh->quads.size());
        for(size_t i = 0;i < res->size();++i){
            const auto& tet = refmesh->_mesh->quads[i];
            pos[i] = zeno::vec3f(0.0);
            Vec12d tet_shape = Vec12d::Zero();
            for(size_t j = 0;j < 4;++j){
                size_t v_id = tet[j];
                const auto& vert = deformation->attr<zeno::vec3f>("pos")[v_id];
                pos[i] += vert;
                tet_shape.segment(j*3,3) << vert[0],vert[1],vert[2];
            }

            pos[i] /= 4;
            Mat3x3d F;
            BaseIntegrator::ComputeDeformationGradient(refmesh->_elmMinv[i],tet_shape,F);
            Mat3x3d U,V;Vec3d s;
            DiffSVD::SVD_Decomposition(F,U,s,V);

            FEM_Scaler vm = compute_von_mises_strain(s);

            es[i] = vm;
        }

        set_output("res",std::move(res));
    }

    FEM_Scaler compute_von_mises_strain(const Vec3d& s) {
        FEM_Scaler s01 = s[0] - s[1];
        FEM_Scaler s02 = s[0] - s[2];
        FEM_Scaler s12 = s[1] - s[2];
        return sqrt(0.5 * (s01*s01 + s02*s02 + s12*s12));
    }
};

ZENDEFNODE(GetEffectiveStrain,{
    {"deform","refMesh"},
    {"res"},
    {},
    {"FEM"},
});

struct GetEffectiveStress : zeno::INode {
    virtual void apply() override {
        auto deformation = get_input<PrimitiveObject>("deform");
        auto refmesh = get_input<FEMMesh>("refMesh");
        auto force_model = get_input<MuscleModelObject>("forceModel");
        auto yield_stress = get_input<zeno::NumericObject>("yield_stress")->get<float>();

        auto res = std::make_shared<PrimitiveObject>();
        auto& pos = res->add_attr<zeno::vec3f>("pos");
        auto& es = res->add_attr<float>("esigma");

        res->resize(refmesh->_mesh->quads.size());
        for(size_t i = 0;i < res->size();++i){
            const auto& tet = refmesh->_mesh->quads[i];
            pos[i] = zeno::vec3f(0.0);
            Vec12d tet_shape = Vec12d::Zero();
            for(size_t j = 0;j < 4;++j){
                size_t v_id = tet[j];
                const auto& vert = deformation->attr<zeno::vec3f>("pos")[v_id];
                pos[i] += vert;
                tet_shape.segment(j*3,3) << vert[0],vert[1],vert[2];
            }

            pos[i] /= 4;
            Mat3x3d F;
            BaseIntegrator::ComputeDeformationGradient(refmesh->_elmMinv[i],tet_shape,F);
            Mat3x3d U,V;Vec3d s;
            DiffSVD::SVD_Decomposition(F,U,s,V);



            TetAttributes attrbs;

            Vec3d ps;
            FEM_Scaler E = refmesh->_elmYoungModulus[i];
            FEM_Scaler nu = refmesh->_elmPossonRatio[i];
            attrbs.emp.E = E;attrbs.emp.nu = nu;attrbs.emp.Act = Mat3x3d::Identity();attrbs.emp.forient << 0.0,1.0,0.0;


            dynamic_cast<ElasticModel*>(force_model->_forceModel.get())->ComputePrincipalStress(attrbs,s,ps);
            FEM_Scaler vm = pow(ps[0] - ps[1],2) + pow(ps[1] - ps[2],2) + pow(ps[0] - ps[2],2);
            vm = vm / 2;
            vm = sqrt(vm);

            es[i] = vm / yield_stress;
        }

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(GetEffectiveStress,{
    {"deform","refMesh","forceModel","yield_stress"},
    {"res"},
    {},
    {"FEM"},
});


struct DebugBSplineImp : zeno::INode {
    virtual void apply() override {
        UniformCubicBasisSpline::DebugCode();
    }
};

ZENDEFNODE(DebugBSplineImp,{
    {},
    {},
    {},
    {"FEM"},
});

struct RetrieveVertIndices : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto bound = get_input<zeno::NumericObject>("bound")->get<float>();

        std::vector<size_t> indices;
        indices.clear();

        const auto& pos = prim->attr<zeno::vec3f>("pos");

        for(size_t i = 0;i < prim->size();++i)
            if(pos[i][1] > bound)
                indices.push_back(i);

        std::cout << "output" << std::endl;
        std::cout << indices.size() << std::endl;
        for(size_t i = 0;i < indices.size();++i)
            std::cout << indices[i] << std::endl;
    }
};

ZENDEFNODE(RetrieveVertIndices,{
    {{"prim"},{"bound"}},
    {},
    {},
    {"FEM"},
});

struct SplineCurve : zeno::IObject {
    SplineCurve() = default;
    std::shared_ptr<UniformCubicBasisSpline> spline;
};

struct MakeNeohookeanSplineModel : zeno::INode {
    virtual void apply() override {
        auto YoungModulus = get_input<zeno::NumericObject>("E")->get<float>();
        auto PossonRatio = get_input<zeno::NumericObject>("nu")->get<float>();


        auto I1Spline = std::make_shared<SplineCurve>();
        auto I2Spline = std::make_shared<SplineCurve>();
        auto I3Spline = std::make_shared<SplineCurve>();

        I1Spline->spline = std::make_shared<UniformCubicBasisSpline>();
        I2Spline->spline = std::make_shared<UniformCubicBasisSpline>();
        I3Spline->spline = std::make_shared<UniformCubicBasisSpline>();

        Vec2d inner_range = Vec2d(0.5,2);
        size_t nm_interps = 6;
        VecXd I1Interps = VecXd::Zero(nm_interps);
        VecXd I2Interps = VecXd::Zero(nm_interps);
        VecXd I3Interps = VecXd::Zero(nm_interps);

        FEM_Scaler inner_width = inner_range[1] - inner_range[0];
        FEM_Scaler step = inner_width / (nm_interps - 1);

        VecXd interp_u = VecXd::Zero(nm_interps);
        for(size_t i = 0;i < nm_interps;++i)
            interp_u[i] = inner_range[0] + step * i;

        FEM_Scaler mu = ElasticModel::Enu2Mu(YoungModulus,PossonRatio);
        FEM_Scaler lambda = ElasticModel::Enu2Lambda(YoungModulus,PossonRatio);

        // neohookean model
        for(size_t i = 0;i < nm_interps;++i)
            I1Interps[i] = 0;
        for(size_t i = 0;i < nm_interps;++i)
            I2Interps[i] = mu / 2 /* + spline define here*/;
        for(size_t i = 0;i < nm_interps;++i){
            I3Interps[i] = -mu + lambda * (interp_u[i] - 1);
            // I3Interps[i] = lambda * (interp_u[i] - 1);
        }

        I1Spline->spline->Interpolate(I1Interps,inner_range);
        I2Spline->spline->Interpolate(I2Interps,inner_range);
        I3Spline->spline->Interpolate(I3Interps,inner_range);

        set_output("I1Spline",I1Spline);
        set_output("I2Spline",I2Spline);
        set_output("I3Spline",I3Spline);
    }
};

ZENDEFNODE(MakeNeohookeanSplineModel,{
    {{"E"},{"nu"}},
    {"I1Spline","I2Spline","I3Spline"},
    {},
    {"FEM"},
});


struct MakeConstantSpline : zeno::INode {
    virtual void apply() override {
        auto constant = get_input<zeno::NumericObject>("constant")->get<float>();
        VecXd yy = VecXd::Zero(5);
        yy.setConstant(constant);
        Vec2d inner_range = Vec2d(0.5,2);
        auto spline = std::make_shared<SplineCurve>();
        spline->spline = std::make_shared<UniformCubicBasisSpline>();
        spline->spline->Interpolate(yy,inner_range);

        set_output("spline",spline);
    }
};

ZENDEFNODE(MakeConstantSpline,{
    {"constant"},
    {"spline"},
    {},
    {"FEM"},
});

struct MakeSplineCurveFromFile : zeno::INode {
    virtual void apply() override {
        auto splineFile = get_input<zeno::StringObject>("SplineFile")->get();
        std::cout << "LOADING SPLINE CURVE FROM : " << splineFile << std::endl;

        std::ifstream fin;
        fin.open(splineFile);
        if(!fin.is_open()){
            std::cerr << "FAILED OPENING FILE " << splineFile << std::endl;
            throw std::runtime_error("FAILED OPENING FILE");
        }
        VecXd xx,yy;

        size_t nm_nodes;
        try{
            fin >> nm_nodes;
            xx.resize(nm_nodes);yy.resize(nm_nodes);
            for(size_t i = 0;i < nm_nodes;++i)
                fin >> xx[i] >> yy[i];
        }catch(const std::exception &e){
            std::cerr << e.what() << std::endl;
        }

        // double step = xx[1] - xx[0];
        // if(step < 0)
        //     throw std::runtime_error("INVALID XX");
        // for(size_t i = 2;i < nm_nodes;++i){
        //     double step_cmp = xx[i] - xx[i-1];
        //     if(fabs(step - step_cmp) > 1e-8){
        //         std::cerr << "ONLY UNIFORM SAMPLED SPLINE IS SUPPORTED" << std::endl;
        //         throw std::runtime_error("ONLY UNIFORM SAMPLED SPLINE IS SUPPORTED");
        //     }
        // }

        Vec2d inner_range = Vec2d(xx[0],xx[nm_nodes - 1]);
        auto spline = std::make_shared<SplineCurve>();
        spline->spline = std::make_shared<UniformCubicBasisSpline>();

        std::cout << "OUTPUT_INTERP: " << std::endl;
        for(size_t i = 0;i < xx.size();++i){
            std::cout << "idx<" << i << "> :\t" << xx[i] << "\t" << yy[i] << std::endl;
        }

        std::cout << "Do the interpolation" << std::endl;
        std::cout << "inner_range : " << inner_range.transpose() << std::endl;
        spline->spline->Interpolate(yy,inner_range);
        std::cout << "finish doing the interpolation" << std::endl;

        set_output("spline",spline);

        std::cout << "finish Loading" << std::endl;
    }
};


ZENDEFNODE(MakeSplineCurveFromFile,{
    {{"readpath","SplineFile"}},
    {"spline"},
    {},
    {"FEM"},
});

struct MakeSplineForceModel : zeno::INode {
    virtual void apply() override {
        auto s1 = get_input<SplineCurve>("S1");
        auto s2 = get_input<SplineCurve>("S2");
        auto s3 = get_input<SplineCurve>("S3");

        auto res = std::make_shared<MuscleModelObject>();

        res->_forceModel = std::shared_ptr<BaseForceModel>(new BSplineIsotropicMuscle(s1->spline,s2->spline,s3->spline));

        set_output("SplineForceModel",res);
    }
};

ZENDEFNODE(MakeSplineForceModel, {
    {"S1","S2","S3"},
    {"SplineForceModel"},
    {},
    {"FEM"},
});

};