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

struct MaterialHandles : zeno::IObject {
    std::vector<std::vector<size_t>> handleIDs;
    std::vector<Vec4d> materials;
};

struct LoadMatertialHandlesFromFile : zeno::INode {
    virtual void apply() override {
        auto file_path = get_input<zeno::StringObject>("handleFile")->get();
        auto outHandles = std::make_shared<MaterialHandles>();
        std::ifstream fin;
        try {
            size_t nm_handles;
            double E,nu,d;
            fin.open(file_path.c_str());
            if (!fin.is_open()) {
                std::cerr << "ERROR::NODE::FAILED::" << file_path << std::endl;
            }
            fin >> nm_handles;
            outHandles->handleIDs.resize(nm_handles);
            outHandles->materials.resize(nm_handles);

            for(size_t i = 0;i < nm_handles;++i){
                size_t nm_vertices;
                double E,nu,d,phi;
                fin >> nm_vertices >> E >> nu >> d >> phi;
                outHandles->materials[i][0] = E;
                outHandles->materials[i][1] = nu;
                outHandles->materials[i][2] = d;
                outHandles->materials[i][3] = phi;

                outHandles->handleIDs[i].resize(nm_vertices);
                for(size_t j = 0;j < nm_vertices;++j)
                    fin >> outHandles->handleIDs[i][j];
            }
            fin.close();
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
        }
        set_output("outHandles",std::move(outHandles));
    }
};

ZENDEFNODE(LoadMatertialHandlesFromFile, {
    {{"readpath","handleFile"}},
    {"outHandles"},
    {},
    {"FEM"},
});

struct FEMMesh : zeno::IObject{
    FEMMesh() = default;
    std::shared_ptr<PrimitiveObject> _mesh;
    std::vector<int> _bouDoFs;
    std::vector<int> _freeDoFs;
    std::vector<int> _DoF2FreeDoF;
    std::vector<int> _SpMatFreeDoFs;
    std::vector<Mat12x12i> _elmSpIndices;

    std::vector<double> _elmMass;
    std::vector<double> _elmVolume;
    std::vector<Mat9x12d> _elmdFdx;
    std::vector<Mat4x4d> _elmMinv;

    std::vector<double> _elmYoungModulus;
    std::vector<double> _elmPossonRatio;
    std::vector<double> _elmDamp;
    std::vector<double> _elmDensity;

    SpMat _connMatrix;
    SpMat _freeConnMatrix;

    std::vector<size_t> _closeBindPoints;
    std::vector<size_t> _farBindPoints;

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
            }
            ele_fin.close();
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
        // std::cout << "nmFree : " << nmFreeDoFs << "\t" << nmDoFs << "\t" << _bouDoFs.size() << std::endl;
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


struct SetMaterialFromHandles : zeno::INode {
    virtual void apply() override {
        auto femmesh = get_input<FEMMesh>("femmesh");
        auto handles = get_input<MaterialHandles>("handles");

        std::vector<Vec4d> materials;
        std::vector<double> weight_sum;

        materials.resize(femmesh->_mesh->size(),Vec4d::Zero());

        double sigma = 0.3;

        #pragma omp parallel for
        for(size_t i = 0;i < materials.size();++i) {
            auto target_vert = femmesh->_mesh->verts[i];
            double weight_sum = 0;
            for(size_t j = 0;j < handles->handleIDs.size();++j){
                Vec4d mat = handles->materials[j];
                const auto& group = handles->handleIDs[j];
                for(size_t k = 0;k < group.size();++k){
                    auto interp_vert = femmesh->_mesh->verts[group[k]];
                    float dissqrt = zeno::lengthSquared(target_vert - interp_vert);
                    float weight = exp(-dissqrt / pow(sigma,2));

                    materials[i] += weight * mat;
                    weight_sum += weight;
                }
            }

            materials[i] /= weight_sum;
        }


        // interpolate the vertex material to element
        size_t nm_elms = femmesh->_mesh->quads.size();
        std::fill(femmesh->_elmYoungModulus.begin(),femmesh->_elmYoungModulus.end(),0);
        std::fill(femmesh->_elmPossonRatio.begin(),femmesh->_elmPossonRatio.end(),0);
        std::fill(femmesh->_elmDamp.begin(),femmesh->_elmDamp.end(),0);
        std::fill(femmesh->_elmDensity.begin(),femmesh->_elmDensity.end(),0);

        #pragma omp parallel for
        for(size_t elm_id = 0;elm_id < femmesh->_mesh->quads.size();++elm_id){
            const auto& elm = femmesh->_mesh->quads[elm_id];
            for(size_t i = 0;i < 4;++i){
                // if(elm_id == 0)
                //     std::cout << "ELM_NU : " << femmesh->_elmPossonRatio[elm_id] << "\t" << materials[elm[i]][1] << std::endl;
                femmesh->_elmYoungModulus[elm_id] += materials[elm[i]][0];
                femmesh->_elmPossonRatio[elm_id] += materials[elm[i]][1];
                femmesh->_elmDamp[elm_id] += materials[elm[i]][2];
                femmesh->_elmDensity[elm_id] += materials[elm[i]][3];
            }

            femmesh->_elmYoungModulus[elm_id] /= 4;
            femmesh->_elmPossonRatio[elm_id] /= 4;
            femmesh->_elmDamp[elm_id] /= 4;
            femmesh->_elmDensity[elm_id] /= 4;
        }

        // std::cout << "output material : " << std::endl;
        // for(size_t i = 0;i < nm_elms;++i)
        //     std::cout << "ELM<" << i << "> : \t" << femmesh->_elmYoungModulus[i] << "\t" \
        //         << femmesh->_elmPossonRatio[i] << "\t" << femmesh->_elmDamp[i] << std::endl;

        set_output("outMesh",femmesh);

        // throw std::runtime_error("material check");
    }
};

ZENDEFNODE(SetMaterialFromHandles, {
    {"femmesh","handles"},
    {"outMesh"},
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

struct FiberParticleToFEMFiber : zeno::INode {
    virtual void apply() override {
        auto fp = get_input<zeno::PrimitiveObject>("fp");
        auto femmesh = get_input<FEMMesh>("femMesh");

        auto fiber = std::make_shared<zeno::PrimitiveObject>();
        auto& fElmID = fiber->add_attr<float>("elmID");
        auto& forient = fiber->add_attr<zeno::vec3f>("orient");
        auto& fact = fiber->add_attr<zeno::vec3f>("act");
        auto& fpos = fiber->add_attr<zeno::vec3f>("pos");
        fiber->resize(femmesh->_mesh->quads.size()); 

        const auto& tets = femmesh->_mesh->quads;  
        const auto& mpos = femmesh->_mesh->verts;

        float sigma = 2;

        #pragma omp parallel for
        for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
            fElmID[elm_id] = float(elm_id);
            auto tet = tets[elm_id];
            fpos[elm_id] = zeno::vec3f(0.0,0.0,0.0);
            for(size_t i = 0;i < 4;++i){
                fpos[elm_id] += mpos[tet[i]];
            }
            fpos[elm_id] /= 4;

            auto& fdir = forient[elm_id];
            fdir = zeno::vec3f(0);
            for(size_t i = 0;i < fp->size();++i){
                const auto& ppos = fp->verts[i];
                const auto& pdir = fp->attr<zeno::vec3f>("vel")[i];

                float dissqrt = zeno::lengthSquared(fpos[elm_id] - ppos);
                float weight = exp(-dissqrt / pow(sigma,2));

                fdir += pdir * weight;
            }
            fdir /= zeno::length(fdir);

            fact[elm_id] = zeno::vec3f(1.0);
        }

        set_output("fiberOut",fiber);     
    }
};

ZENDEFNODE(FiberParticleToFEMFiber, {
    {"fp","femMesh"},
    {"fiberOut"},
    {},
    {"FEM"}
});

struct FiberToFiberSegements : zeno::INode {
    virtual void apply() override {
        auto fiber = get_input<zeno::PrimitiveObject>("fibers");
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        const auto& mpos = fiber->verts;
        const auto& mvel = fiber->attr<zeno::vec3f>("vel");

        auto fiberSeg = std::make_shared<zeno::PrimitiveObject>();
        fiberSeg->resize(fiber->size() * 2);
        auto& flines = fiberSeg->lines;
        flines.resize(fiber->size());
        auto& fpos = fiberSeg->verts;

        for(size_t i = 0;i < fiber->size();++i){
            flines[i] = zeno::vec2i(i,i + fiber->size());
            fpos[i] = mpos[i];
            fpos[i + fiber->size()] = fpos[i] + dt * mvel[i];
        }

        set_output("fseg",fiberSeg);
    }
};

ZENDEFNODE(FiberToFiberSegements, {
    {"fibers","dt"},
    {"fseg"},
    {},
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

struct PrimitieveOut : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        set_output("pout",prim);
    }
};

ZENDEFNODE(PrimitieveOut, {
    {"pin"},
    {"pout"},
    {},
    {"FEM"},
});


struct MakeFEMMeshFromFile : zeno::INode{
    virtual void apply() override {
        auto node_file = get_input<zeno::StringObject>("NodeFile")->get();
        auto ele_file = get_input<zeno::StringObject>("EleFile")->get();    
        // auto bou_file = get_input<zeno::StringObject>("BouFile")->get();
        auto bind_file = get_input<zeno::StringObject>("BouFile")->get();
        float density = get_input<zeno::NumericObject>("density")->get<float>();
        float E = get_input<zeno::NumericObject>("YoungModulus")->get<float>();
        float nu = get_input<zeno::NumericObject>("PossonRatio")->get<float>();
        float d = get_input<zeno::NumericObject>("Damp")->get<float>();

        auto res = std::make_shared<FEMMesh>();
        res->_mesh = std::make_shared<PrimitiveObject>();

        res->LoadVerticesFromFile(node_file);
        res->LoadElementsFromFile(ele_file);
        // res->LoadBoundaryIndicesFromFile(bou_file);
        std::cout << "load bou vertices" << std::endl;
        // res->LoadBoundaryVerticesFromFile(bou_file);
        res->LoadBindingPoints(bind_file);

        std::cout << "finish loading bind vertices" << std::endl;
        size_t nm_con_vertices = res->_closeBindPoints.size() + res->_farBindPoints.size();
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

        // std::cout << "cons_dofs : " << std::endl;
        // for(size_t i = 0;i < res->_bouDoFs.size();++i)
        //     std::cout << i << "\t" << res->_bouDoFs[i] << std::endl;

        // throw std::runtime_error("check");

        for(size_t i = 0;i < res->_mesh->quads.size();++i){
            auto tet = res->_mesh->quads[i];
            res->_mesh->tris.emplace_back(tet[0],tet[1],tet[2]);
            res->_mesh->tris.emplace_back(tet[1],tet[3],tet[2]);
            res->_mesh->tris.emplace_back(tet[0],tet[2],tet[3]);
            res->_mesh->tris.emplace_back(tet[0],tet[3],tet[1]);
        }
// allocate memory
        size_t nm_elms = res->_mesh->quads.size();
        res->_elmYoungModulus.resize(nm_elms,E);
        res->_elmPossonRatio.resize(nm_elms,nu);
        res->_elmDamp.resize(nm_elms,d);
        res->_elmDensity.resize(nm_elms,density);

        res->_elmVolume.resize(nm_elms);
        res->_elmdFdx.resize(nm_elms);
        res->_elmMass.resize(nm_elms);
        res->_elmMinv.resize(nm_elms);

        std::cout << "updating dofs map" << std::endl;
        res->UpdateDoFsMapping();
        std::cout << "finish updating dofs map" << std::endl;
        res->DoPreComputation();
        std::cout << "finish precompuation" << std::endl;
// rendering mesh
        auto resGeo = std::make_shared<PrimitiveObject>();
        auto &pos = resGeo->add_attr<zeno::vec3f>("pos");
        // std::cout << "OUTPUT FRAME " << cur_frame.norm() << std::endl;
        for(size_t i = 0;i < res->_mesh->size();++i){
            auto vert = res->_mesh->verts[i];
            pos.emplace_back(vert[0],vert[1],vert[2]);
        }

        for(int i=0;i < res->_mesh->quads.size();++i){
            auto tet = res->_mesh->quads[i];
            resGeo->tris.emplace_back(tet[0],tet[1],tet[2]);
            resGeo->tris.emplace_back(tet[1],tet[3],tet[2]);
            resGeo->tris.emplace_back(tet[0],tet[2],tet[3]);
            resGeo->tris.emplace_back(tet[0],tet[3],tet[1]);
        }
        pos.resize(res->_mesh->size());

        set_output("FEMMesh",res);

        std::cout << "finish loading fem mesh" << std::endl;
    }
};

ZENDEFNODE(MakeFEMMeshFromFile, {
    {{"readpath","NodeFile"},{"readpath", "EleFile"},{"readpath","BouFile"},
        {"density"},{"YoungModulus"},{"PossonRatio"},{"Damp"}},
    {"FEMMesh"},
    {},
    {"FEM"},
});

struct SetUniformActivation : zeno::INode {
    virtual void apply() override {
        auto fiber = get_input<PrimitiveObject>("fiber");
        auto act = get_input<zeno::NumericObject>("uniform_act")->get<zeno::vec3f>();
        for(size_t i = 0;i < fiber->size();++i)
            fiber->attr<zeno::vec3f>("act")[i] = act;
        set_output("fiberOut",fiber);
    }
};

ZENDEFNODE(SetUniformActivation, {
    {{"fiber"},{"uniform_act"}},
    {"fiberOut"},
    {},
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

    VecXd& GetCurrentFrame() {return _traj[(_stepID + _intPtr->GetCouplingLength()) % _intPtr->GetCouplingLength()];} 

    void AssignElmAttribs(size_t elm_id,TetAttributes& attrbs,const std::shared_ptr<FEMMesh>& mesh,const std::shared_ptr<PrimitiveObject>& fiber,const Vec12d& ext_force) const {
        attrbs._elmID = elm_id;
        attrbs._Minv = mesh->_elmMinv[elm_id];
        attrbs._dFdX = mesh->_elmdFdx[elm_id];
        if(fiber){
            const auto& orient = fiber->attr<zeno::vec3f>("orient")[elm_id];
            attrbs.emp.forient << orient[0],orient[1],orient[2];
            zeno::vec3f act = fiber->attr<zeno::vec3f>("act")[elm_id];
            Vec3d act_vec;act_vec << act[0],act[1],act[2];
            Mat3x3d R = MatHelper::Orient2R(attrbs.emp.forient);
            attrbs.emp.Act = R * act_vec.asDiagonal() * R.transpose();
        }else{
            attrbs.emp.forient << 1.0,0.0,0.0;
            attrbs.emp.Act =  Mat3x3d::Identity();
        }

        attrbs.emp.E = mesh->_elmYoungModulus[elm_id];
        attrbs.emp.nu = mesh->_elmPossonRatio[elm_id];
        attrbs.v = mesh->_elmDamp[elm_id];
        attrbs._volume = mesh->_elmVolume[elm_id];
        attrbs._density = mesh->_elmDensity[elm_id];
        attrbs._ext_f = ext_force; 
    }


    FEM_Scaler EvalObj(const std::shared_ptr<FEMMesh>& mesh,
        const std::shared_ptr<PrimitiveObject>& fiber,
        const std::shared_ptr<MuscleModelObject>& muscle,
        const std::shared_ptr<DampingForceModel>& damp) {
            FEM_Scaler obj = 0;
            size_t nm_elms = mesh->_mesh->quads.size();

            _objBuffer.resize(nm_elms);
            
            size_t clen = _intPtr->GetCouplingLength();

            // #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                auto tet = mesh->_mesh->quads[elm_id];

                Vec12d elm_ext_force;
                RetrieveElmVector(tet,elm_ext_force,_extForce);
                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh,fiber,elm_ext_force);

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
        const std::shared_ptr<PrimitiveObject>& fiber,
        const std::shared_ptr<MuscleModelObject>& muscle,
        const std::shared_ptr<DampingForceModel>& damp,
        VecXd& deriv) {
            FEM_Scaler obj = 0;
            size_t nm_elms = mesh->_mesh->quads.size();

            size_t clen = _intPtr->GetCouplingLength();

            _objBuffer.resize(nm_elms);
            _derivBuffer.resize(nm_elms);
            
            // #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];

                Vec12d elm_ext_force;
                RetrieveElmVector(tet,elm_ext_force,_extForce);

                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh,fiber,elm_ext_force);

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
        const std::shared_ptr<PrimitiveObject>& fiber,
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

            // #pragma omp parallel for 
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                // std::cout << "attrbs.elm_Act" << std::endl << attrbs._activation << std::endl;
                // throw std::runtime_error("CHECK ATTRS");
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];

                Vec12d elm_ext_force;
                RetrieveElmVector(tet,elm_ext_force,_extForce);

                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh,fiber,elm_ext_force);                

                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],_traj[frameID]);
                }

                _intPtr->EvalElmObjDerivJacobi(attrbs,
                    muscle->_forceModel,
                    damp->_dampForce,
                    elm_traj,
                    &_objBuffer[elm_id],_derivBuffer[elm_id],_HBuffer[elm_id],enforce_spd);
            }

            deriv.setZero();
            HValBuffer.setZero();
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                auto tet = mesh->_mesh->quads[elm_id];
                obj += _objBuffer[elm_id];
                AssembleElmVector(tet,_derivBuffer[elm_id],deriv);
                AssembleElmMatrixAdd(tet,_HBuffer[elm_id],mesh->MapHMatrixRef(HValBuffer.data()));

            }
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
        auto mesh = get_input<FEMMesh>("ref_mesh");
        auto integrator = get_input<FEMIntegrator>("intIn");
        const auto& frame = integrator->GetCurrentFrame();

        auto resGeo = std::make_shared<PrimitiveObject>();
        auto &pos = resGeo->add_attr<zeno::vec3f>("pos");
        // std::cout << "OUTPUT FRAME " << cur_frame.norm() << std::endl;
        for(size_t i = 0;i < mesh->_mesh->size();++i){
            auto vert = frame.segment(i*3,3);
            pos.emplace_back(vert[0],vert[1],vert[2]);
        }
        for(int i=0;i < mesh->_mesh->quads.size();++i){
            auto tet = mesh->_mesh->quads[i];
            resGeo->tris.emplace_back(tet[0],tet[1],tet[2]);
            resGeo->tris.emplace_back(tet[1],tet[3],tet[2]);
            resGeo->tris.emplace_back(tet[0],tet[2],tet[3]);
            resGeo->tris.emplace_back(tet[0],tet[3],tet[1]);
        }

        set_output("frame",std::move(resGeo));
    }
};


ZENDEFNODE(GetCurrentFrame,{
    {{"ref_mesh"},{"intIn"}},
    {"frame"},
    {},
    {"FEM"},
});

struct MakeFEMIntegrator : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("Mesh");
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
                    VecXd::Zero(mesh->_mesh->size() * 3));
        for(size_t i = 0;i < res->_intPtr->GetCouplingLength();++i)
            for(size_t j = 0;j < mesh->_mesh->size();++j){
                auto& vert = mesh->_mesh->verts[j];
                res->_traj[i][j*3 + 0] = vert[0];
                res->_traj[i][j*3 + 1] = vert[1];
                res->_traj[i][j*3 + 2] = vert[2];
            }

        res->_stepID = 0;

        res->_extForce.resize(mesh->_mesh->size() * 3);
        res->_objBuffer.resize(mesh->_mesh->quads.size());
        res->_derivBuffer.resize(mesh->_mesh->quads.size());

        set_output("FEMIntegrator",res);
    }
};

ZENDEFNODE(MakeFEMIntegrator,{
    {{"Mesh"},{"gravity"},{"dt"}},
    {"FEMIntegrator"},
    {{"enum BackwardEuler QuasiStatic", "integType", "BackwardEuler"}},
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

struct SetBoundaryMotion : zeno::INode {
       virtual void apply() override {
        auto intPtr = get_input<FEMIntegrator>("intIn");
        auto mesh = get_input<FEMMesh>("refMesh");
        auto CT = get_input<TransformMatrix>("CT");
        auto FT = get_input<TransformMatrix>("FT");

        for(size_t i = 0;i < mesh->_closeBindPoints.size();++i){
            size_t vertID = mesh->_closeBindPoints[i];
            const auto& ref_vert = mesh->_mesh->attr<zeno::vec3f>("pos")[vertID];
            Vec4d vert_w;vert_w << ref_vert[0],ref_vert[1],ref_vert[2],1.0;
            vert_w = CT->Mat * vert_w;
            intPtr->GetCurrentFrame().segment(vertID*3,3) = vert_w.segment(0,3);
        }

        for(size_t i = 0;i < mesh->_farBindPoints.size();++i){
            size_t vertID = mesh->_farBindPoints[i];
            const auto& ref_vert = mesh->_mesh->attr<zeno::vec3f>("pos")[vertID];
            Vec4d vert_w;vert_w << ref_vert[0],ref_vert[1],ref_vert[2],1.0;
            vert_w = FT->Mat * vert_w;
            intPtr->GetCurrentFrame().segment(vertID*3,3) = vert_w.segment(0,3);
        }

        set_output("intOut",intPtr);   
    } 
};

ZENDEFNODE(SetBoundaryMotion,{
    {{"intIn"},{"refMesh"},{"CT"},{"FT"}},
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
    std::vector<size_t> _closedBindPoints;
    std::vector<size_t> _farBindPoints;
};

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

        std::shared_ptr<PrimitiveObject> fiber = nullptr;
        if(has_input("fiber")){
            fiber = get_input<PrimitiveObject>("fiber");
        }

        int max_iters = get_param<int>("maxNRIters");
        int max_linesearch = get_param<int>("maxBTLs");
        float c1 = get_param<float>("ArmijoCoeff");
        float c2 = get_param<float>("CurvatureCoeff");
        float beta = get_param<float>("BTL_shrinkingRate");
        float epsilon = get_param<float>("epsilon");

        // the wolfe condtion buffer mainly for debugging
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
        do{
            FEM_Scaler e0,e1,eg0;

            e0 = integrator->EvalObjDerivHessian(mesh,fiber,force_model,damping_model,r,HBuffer,true);

            MatHelper::RetrieveDoFs(r.data(),ruc.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());
            MatHelper::RetrieveDoFs(HBuffer.data(),HucBuffer.data(),mesh->_SpMatFreeDoFs.size(),mesh->_SpMatFreeDoFs.data());

            if(ruc.norm() < epsilon){
                std::cout << "BREAK WITH RUC = " << ruc.norm() << std::endl;
                break;
            }
            ruc *= -1;

            if(!analyized_pattern){
                _LDLTSolver.analyzePattern(mesh->MapHucMatrix(HucBuffer.data()));
                analyized_pattern = true;
            }

            _LDLTSolver.factorize(mesh->MapHucMatrix(HucBuffer.data()));
            dpuc = _LDLTSolver.solve(ruc);

            eg0 = -dpuc.dot(ruc);

            if(fabs(eg0) < epsilon * epsilon){
                std::cout << "BREAK WITH EG0 = " << eg0 << "\t" << ruc.norm() << std::endl;
                break;
            }

            if(eg0 > 0){
                throw std::runtime_error("non-negative descent direction");
            }
            dp.setZero();
            MatHelper::UpdateDoFs(dpuc.data(),dp.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());

            search_idx = 0;

            FEM_Scaler alpha = 2.0f;
            FEM_Scaler beta = 0.5f;
            FEM_Scaler c1 = 0.01f;

            double armijo_condition;
            do{
                if(search_idx != 0)
                    integrator->GetCurrentFrame() -= alpha * dp;
                alpha *= beta;
                integrator->GetCurrentFrame() += alpha * dp;
                e1 = integrator->EvalObj(mesh,fiber,force_model,damping_model);
                ++search_idx;
                wolfeBuffer[search_idx-1](0) = (e1 - e0)/alpha;
                wolfeBuffer[search_idx-1](1) = eg0;

                armijo_condition = double(e1) - double(e0) - double(c1)*double(alpha)*double(eg0);
            }while(/*(e1 > e0 + c1*alpha*eg0)*/ armijo_condition > 0.0f /* || (fabs(eg1) > c2*fabs(eg0))*/ && (search_idx < max_linesearch));

            if(search_idx == max_linesearch){
                std::cout << "LINESEARCH EXCEED" << std::endl;
                for(size_t i = 0;i < max_linesearch;++i)
                    std::cout << "idx:" << i << "\t" << wolfeBuffer[i].transpose() << std::endl;

                
                throw std::runtime_error("LINESEARCH");
            }
            ++iter_idx;
        }while(iter_idx < max_iters);

        if(iter_idx == max_iters){
            std::cout << "MAX NEWTON ITERS EXCEED" << std::endl;
        }
        set_output("intOut",std::move(integrator));     
    }
};

ZENDEFNODE(SolveEquaUsingNRSolver,{
    {{"mesh","muscleForce","dampForce","integrator"}},
    {"intOut"},
    {{"int","maxNRIters","10"},{"int","maxBTLs","10"},{"float","ArmijoCoeff","0.01"},{"float","CurvatureCoeff","0.9"},{"float","BTL_shrinkingRate","0.5"},{"float","epsilon","1e-6"}},
    {"FEM"},
});

struct GetEffectiveStrain : INode {
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