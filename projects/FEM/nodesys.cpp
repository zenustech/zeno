#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

// #include <anisotropic_NH.h>
#include <diriclet_damping.h>
#include <stable_isotropic_NH.h>

#include <backward_euler_integrator.h>
#include <fstream>
#include <algorithm>

#include <matrix_helper.hpp>
#include<Eigen/SparseCholesky>
#include <iomanip>

#include <cmath>

#include "matrixObject.h"

namespace{

using namespace zeno;

struct MaterialHandles : zeno::IObject {
    std::vector<std::vector<size_t>> handleIDs;
    std::vector<Vec3d> materials;
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
                double E,nu,d;
                fin >> nm_vertices >> E >> nu >> d;
                outHandles->materials[i][0] = E;
                outHandles->materials[i][1] = nu;
                outHandles->materials[i][2] = d;

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

    std::vector<Mat3x3d> _elmAct;
    std::vector<Mat3x3d> _elmOrient;
    std::vector<Vec3d> _elmWeight;

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

// load .ele file
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
            // std::cout << "nm_con_vertices : " << nm_con_vertices << std::endl;
            // std::cout << "start_idx : " << start_idx << std::endl;
            // std::cout << "filename : " << filename << std::endl;
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

            // std::cout << "cons_dofs : " << std::endl;
            // for(size_t i = 0;i < _bouDoFs.size();++i)
                // std::cout << i << "\t" << _bouDoFs[i] << std::endl;

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

        // std::cout << "check point 1" << std::endl;

        // std::cout << "nm_bou_dofs : "<< nmBouDoFs << std::endl;
        // for(size_t i = 0;i < nmBouDoFs;++i){
        //     std::cout << _bouDoFs[i] << std::endl;
        // }

        // std::cout << "nm_dofs : " << nmDoFs << std::endl;
        // std::cout << "max_cdof : " << _bouDoFs[nmBouDoFs - 1] << std::endl;

        // std::cout << "END" << std::endl;

        for(size_t cdof_idx = 0,dof = 0,ucdof_count = 0;dof < nmDoFs;++dof){
            if(cdof_idx >= nmBouDoFs || dof != _bouDoFs[cdof_idx]){
                // std::cout << "cdof_idx  = " << cdof_idx << "\t" << 
                _freeDoFs[ucdof_count] = dof;
                ++ucdof_count;
            }
            else
                ++cdof_idx;
        }

        // std::cout << "check point 2 dofs = " << nmDoFs << std::endl;
// build uc mapping
        _DoF2FreeDoF.resize(nmDoFs);
        // std::cout << "nmDoFs : " << nmDoFs << std::endl;
        std::fill(_DoF2FreeDoF.begin(),_DoF2FreeDoF.end(),-1);
        // std::cout << "..." << std::endl;
        for(size_t i = 0;i < _freeDoFs.size();++i){
            int ucdof = _freeDoFs[i];
            _DoF2FreeDoF[ucdof] = i;
        }

        // std::cout << "check point 3" << std::endl;
// Initialize connectivity matrices
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
        // std::cout << "check point 4" << std::endl;
        _connMatrix = SpMat(nmDoFs,nmDoFs);
        // std::cout << "check point 5" << std::endl;

        // size_t iter_idx = 0;
        // for(auto iter = connTriplets.begin();iter != connTriplets.end();++iter,iter_idx++){
        //     std::cout << "T<" << iter_idx  << "> : " << iter->row() << "\t" << iter->col() << "\t" << iter->value() << std::endl;
        //     if(iter_idx > 100)
        //         break;
        // }

        _connMatrix.setFromTriplets(connTriplets.begin(),connTriplets.end());
        
        // std::cout << "check point 6" << std::endl;
        _connMatrix.makeCompressed();

        // std::cout << "constrained dofs : '" << std::endl;
        // for(size_t i = 0;i < )

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

        // std::cout << "check point 5" << std::endl;

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

        // std::cout << "check point 6" << std::endl;
    }
};


struct SetMaterialFromHandles : zeno::INode {
    virtual void apply() override {
        auto femmesh = get_input<FEMMesh>("femmesh");
        auto handles = get_input<MaterialHandles>("handles");

        std::vector<Vec3d> materials;
        std::vector<double> weight_sum;

        materials.resize(femmesh->_mesh->size(),Vec3d::Zero());

        double sigma = 0.3;

        #pragma omp parallel for
        for(size_t i = 0;i < materials.size();++i) {
            auto target_vert = femmesh->_mesh->verts[i];
            double weight_sum = 0;
            for(size_t j = 0;j < handles->handleIDs.size();++j){
                Vec3d mat = handles->materials[j];
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

        #pragma omp parallel for
        for(size_t elm_id = 0;elm_id < femmesh->_mesh->quads.size();++elm_id){
            const auto& elm = femmesh->_mesh->quads[elm_id];
            for(size_t i = 0;i < 4;++i){
                // if(elm_id == 0)
                //     std::cout << "ELM_NU : " << femmesh->_elmPossonRatio[elm_id] << "\t" << materials[elm[i]][1] << std::endl;
                femmesh->_elmYoungModulus[elm_id] += materials[elm[i]][0];
                femmesh->_elmPossonRatio[elm_id] += materials[elm[i]][1];
                femmesh->_elmDamp[elm_id] += materials[elm[i]][2];
            }

            femmesh->_elmYoungModulus[elm_id] /= 4;
            femmesh->_elmPossonRatio[elm_id] /= 4;
            femmesh->_elmDamp[elm_id] /= 4;
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
        fiber->add_attr<float>("elmID");
        fiber->add_attr<zeno::vec3f>("vel");
        fiber->resize(femmesh->_mesh->quads.size()); 

        const auto& tets = femmesh->_mesh->quads;  
        const auto& mpos = femmesh->_mesh->verts;

        float sigma = 2;


        #pragma omp parallel for
        for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
            fiber->attr<float>("elmID")[elm_id] = float(elm_id);

            auto tet = tets[elm_id];
            auto& fpos = fiber->attr<zeno::vec3f>("pos")[elm_id];
            for(size_t i = 0;i < 4;++i){
                fpos += mpos[tet[i]];
            }
            fpos /= 4;

            auto& fdir = fiber->attr<zeno::vec3f>("vel")[elm_id];
            fdir = zeno::vec3f(0);
            for(size_t i = 0;i < fp->size();++i){
                const auto& ppos = fp->verts[i];
                const auto& pdir = fp->attr<zeno::vec3f>("vel")[i];

                float dissqrt = zeno::lengthSquared(fpos - ppos);
                float weight = exp(-dissqrt / pow(sigma,2));

                fdir += pdir * weight;
            }

            fdir /= zeno::length(fdir);
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


struct FEMMAddFibers : zeno::INode {
    virtual void apply() override {
        const auto& fibers = get_input<zeno::PrimitiveObject>("fibers");
        auto femmesh = get_input<FEMMesh>("inMesh");
        auto fweight = get_input<zeno::NumericObject>("weight")->get<zeno::vec3f>();
        
        assert(fibers->size() == femmesh->_mesh->quads.size());

        const auto& fposs = fibers->attr<zeno::vec3f>("pos");
        const auto& fdirs = fibers->attr<zeno::vec3f>("vel");
        const auto& elmIDs = fibers->attr<float>("elmID");

        const auto& mtets = femmesh->_mesh->quads;
        const auto& mverts = femmesh->_mesh->verts;
        auto& mOrients = femmesh->_elmOrient;
        auto& mWeights = femmesh->_elmWeight;

        for(size_t elm_id = 0;elm_id < femmesh->_mesh->quads.size();++elm_id){
            assert(elmIDs[elm_id] == elm_id);
            mWeights[elm_id] << fweight[0],fweight[1],fweight[2];

            auto dir0 = fdirs[elm_id] / zeno::length(fdirs[elm_id]);
            auto tmp_dir = dir0;
            tmp_dir[0] += 1;
            auto dir1 = zeno::cross(dir0,tmp_dir);
            dir1 /= zeno::length(dir1);
            auto dir2 = zeno::cross(dir0,dir1);
            dir2 /= zeno::length(dir2);

            Mat3x3d orient;
            orient.col(0) << dir0[0],dir0[1],dir0[2];
            orient.col(1) << dir1[0],dir1[1],dir1[2];
            orient.col(2) << dir2[0],dir2[1],dir2[2];

            mOrients[elm_id] = orient;
        }

        set_output("outMesh",femmesh);

    }
};

ZENDEFNODE(FEMMAddFibers, {
    {"fibers","inMesh","weight"},
    {"outMesh"},
    {},
    {"FEM"},
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



struct AddMuscleFibers : zeno::INode {
    virtual void apply() override {
        auto fibers = get_input<zeno::PrimitiveObject>("fibers");
        auto femmesh = get_input<FEMMesh>("femmesh");
        auto fposs = fibers->attr<zeno::vec3f>("pos");
        auto fdirs = fibers->attr<zeno::vec3f>("vel");

        const auto &mpos = femmesh->_mesh->attr<zeno::vec3f>("pos");
        const auto &tets = femmesh->_mesh->quads;

        std::vector<zeno::vec3f> tet_dirs;
        std::vector<zeno::vec3f> tet_pos;
        tet_dirs.resize(tets.size(),zeno::vec3f(0));
        tet_pos.resize(tets.size());

        // Retrieve the center of the tets
        for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
            auto tet = tets[elm_id];
            tet_pos[elm_id] = zeno::vec3f(0);

            for(size_t i = 0;i < 4;++i){
                tet_pos[elm_id] += mpos[tet[i]];
            }
            tet_pos[elm_id] /= 4;
        }

        float sigma = 2;

        for(size_t i = 0;i < fibers->size();++i){
            auto fpos = fposs[i];
            auto fdir = fdirs[i];
            fdir /= zeno::length(fdir);

            for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
                float dissqrt = zeno::lengthSquared(fpos - tet_pos[elm_id]);
                float weight = exp(-dissqrt / pow(sigma,2));
                tet_dirs[elm_id] += weight * fdir;
            }
        }

        for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
            tet_dirs[elm_id] /= zeno::length(tet_dirs[elm_id]);
        }

        for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
            auto dir0 = tet_dirs[elm_id] / zeno::length(tet_dirs[elm_id]);
            auto ref_dir = dir0;
            ref_dir[0] += 1;
            auto dir1 = zeno::cross(dir0,ref_dir);
            dir1 /= zeno::length(dir1);
            auto dir2 = zeno::cross(dir0,dir1);
            dir2 /= zeno::length(dir2);

            Mat3x3d orient;
            orient.col(0) << dir0[0],dir0[1],dir0[2];
            orient.col(1) << dir1[0],dir1[1],dir1[2];
            orient.col(2) << dir2[0],dir2[1],dir2[2];

            femmesh->_elmOrient[elm_id] = orient;
        }

        set_output("outMesh",femmesh);

        std::cout << "output fiber geo" << std::endl;

        auto fiber = std::make_shared<zeno::PrimitiveObject>();
        fiber->add_attr<float>("elmID");
        fiber->add_attr<zeno::vec3f>("int_weight");
        fiber->resize(tets.size() * 2);

        float length = 400;
        if(has_input("length")){
            length = get_input<NumericObject>("length")->get<float>();
        }

        auto& pos = fiber->attr<zeno::vec3f>("pos");
        auto& lines = fiber->lines;
        auto& elmIDs = fiber->attr<float>("elmID");
        auto& intWeights = fiber->attr<zeno::vec3f>("int_weight");

        lines.resize(tets.size());
        float dt = 0.01;

        for(size_t elm_id = 0;elm_id < tets.size();++elm_id){
            pos[elm_id] = tet_pos[elm_id];
            auto pend = tet_pos[elm_id] + dt * tet_dirs[elm_id] * length;
            pos[elm_id + tets.size()] = pend;
            lines[elm_id] = zeno::vec2i(elm_id,elm_id + tets.size());

            elmIDs[elm_id] = elm_id;
            elmIDs[elm_id + tets.size()] = elm_id;

            intWeights[elm_id] = zeno::vec3f(0.25,0.25,0.25);


        }

        set_output("fiberGeo",fiber);

    }
};

ZENDEFNODE(AddMuscleFibers, {
    {"fibers","femmesh"},
    {"outMesh","fiberGeo"},
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
        res->_elmAct.resize(nm_elms,Mat3x3d::Identity());
        res->_elmOrient.resize(nm_elms,Mat3x3d::Identity());
        res->_elmWeight.resize(nm_elms,Vec3d(1.0,0.5,0.5));
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

struct SetUniformMuscleAnisotropicWeight : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("inputMesh");
        auto uni_weight = get_input<zeno::NumericObject>("weight")->get<zeno::vec3f>();
        for(size_t i = 0;i < mesh->_mesh->quads.size();++i){
            mesh->_elmWeight[i] << uni_weight[0],uni_weight[1],uni_weight[2];
        }
        set_output("aniMesh",mesh);
    }
};

ZENDEFNODE(SetUniformMuscleAnisotropicWeight, {
    {{"inputMesh"},{"weight"}},
    {"aniMesh"},
    {},
    {"FEM"},
});

struct SetUniformActivation : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("inputMesh");
        auto uniform_Act = get_input<zeno::NumericObject>("uniform_act")->get<zeno::vec3f>();

        for(size_t i = 0;i < mesh->_mesh->quads.size();++i){
            Mat3x3d fdir = mesh->_elmOrient[i];
            Vec3d act_vec;
            act_vec << uniform_Act[0],uniform_Act[1],uniform_Act[2]; 
            mesh->_elmAct[i] << fdir * act_vec.asDiagonal() * fdir.transpose();
        }

        set_output("actMesh",mesh);
    }
};

ZENDEFNODE(SetUniformActivation, {
    {{"inputMesh"},{"uniform_act"}},
    {"actMesh"},
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


struct MuscleForceModel : zeno::IObject {
    MuscleForceModel() = default;
    std::shared_ptr<BaseForceModel> _forceModel;
};

struct DampingForceModel : zeno::IObject {
    DampingForceModel() = default;
    std::shared_ptr<DiricletDampingModel> _dampForce;
};

struct MakeMuscleForceModel : zeno::INode {
    virtual void apply() override {
        auto model_type = std::get<std::string>(get_param("ForceModel"));
        auto res = std::make_shared<MuscleForceModel>();
        if(model_type == "Fiberic"){
            // res->_forceModel = std::shared_ptr<BaseForceModel>(new AnisotropicSNHModel());
            std::cout << "The Anisotropic Model is not stable yet" << std::endl;
            throw std::runtime_error("The Anisotropic Model is not stable yet");
        }
        else if(model_type == "HyperElastic")
            res->_forceModel = std::shared_ptr<BaseForceModel>(new StableIsotropicMuscle());
        else{
            std::cerr << "UNKNOWN MODEL_TYPE" << std::endl;
            throw std::runtime_error("UNKNOWN MODEL_TYPE");
        }
        
        set_output("MuscleForceModel",res);
    }
};

ZENDEFNODE(MakeMuscleForceModel, {
    {},
    {"MuscleForceModel"},
    {{"enum HyperElastic Fiberic", "ForceModel", "HyperElastic"}},
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
    std::shared_ptr<BackEulerIntegrator> _intPtr;
    std::vector<VecXd> _traj;
    size_t _stepID;
};

struct MakeFEMIntegrator : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("Mesh");
        auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();

        auto res = std::make_shared<FEMIntegrator>();
        res->_intPtr = std::make_shared<BackEulerIntegrator>();
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

        set_output("FEMIntegrator",res);
    }
};

ZENDEFNODE(MakeFEMIntegrator,{
    {{"Mesh"},{"gravity"},{"dt"}},
    {"FEMIntegrator"},
    {},
    {"FEM"},
});

struct SetInitialDeformation : zeno::INode {
    virtual void apply() override {
        auto integrator = get_input<FEMIntegrator>("integrator");
        auto deformation = get_input<zeno::NumericObject>("deform")->get<zeno::vec3f>();
        for(size_t i = 0;i < integrator->_intPtr->GetCouplingLength();++i){
            for(size_t j = 0;j < integrator->_traj[0].size()/3;++j){
                integrator->_traj[i][j*3 + 0] *= deformation[0];
                integrator->_traj[i][j*3 + 0] *= deformation[1];
                integrator->_traj[i][j*3 + 0] *= deformation[2];
            }
        }

        integrator->_stepID = 0;
        set_output("intOut",integrator);
    }
};

ZENDEFNODE(SetInitialDeformation,{
    {{"integrator"},{"deform"}},
    {"intOut"},
    {},
    {"FEM"},
});




struct RetrieveRigidTransform : zeno::INode {
    virtual void apply() override {
        // std::cout << "AAAA" << std::endl;

        auto objRef = get_input<zeno::PrimitiveObject>("refObj");
        auto objNew = get_input<zeno::PrimitiveObject>("newObj");

        Mat4x4d refTet,newTet;
        for(size_t i = 0;i < 4;++i){
            refTet.col(i) << objRef->verts[i][0],objRef->verts[i][1],objRef->verts[i][2],1.0;
            newTet.col(i) << objNew->verts[i][0],objNew->verts[i][1],objNew->verts[i][2],1.0;
        }

        Mat4x4d T = newTet * refTet.inverse();

        std::cout << "T : " << std::endl << T << std::endl; 

        auto ret = std::make_shared<TransformMatrix>();
        ret->Mat = T;

        set_output("T",std::move(ret));
        // std::cout << "BBBB" << std::endl;
    }
};

ZENDEFNODE(RetrieveRigidTransform,{
    {{"refObj"},{"newObj"}},
    {"T"},
    {},
    {"FEM"},
});

struct DoTimeStep : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<FEMMesh>("mesh");
        auto force_model = get_input<MuscleForceModel>("muscleForce");
        auto damping_model = get_input<DampingForceModel>("dampForce");
        auto integrator = get_input<FEMIntegrator>("integrator");
        auto epsilon = get_input<zeno::NumericObject>("epsilon")->get<float>();
        auto closed_T = get_input<TransformMatrix>("CT");
        auto far_T = get_input<TransformMatrix>("FT");

        size_t clen = integrator->_intPtr->GetCouplingLength();
        size_t curID = (integrator->_stepID + clen) % clen;
        size_t preID = (integrator->_stepID + clen - 1) % clen;

        // set initial guess
        integrator->_traj[curID] = integrator->_traj[preID];

        // if(integrator->_stepID == 1){
        //     std::cout << "materials : " << std::endl;
        //     for(size_t i = 0;i < mesh->_mesh->quads.size();++i){
        //         std::cout << "M<" << i << "> : " << mesh->_elmYoungModulus[i] << "\t" << mesh->_elmPossonRatio[i] << "\t" << mesh->_elmDamp[i] << std::endl;
        //     }
        // }

        auto depa = std::make_shared<zeno::PrimitiveObject>();
        auto& depa_pos = depa->attr<zeno::vec3f>("pos");

        for(size_t i = 0;i < mesh->_closeBindPoints.size();++i){
            size_t idx = mesh->_closeBindPoints[i];
            Vec4d vert;
            vert << mesh->_mesh->verts[idx][0],mesh->_mesh->verts[idx][1],mesh->_mesh->verts[idx][2],1.0;
            vert = closed_T->Mat * vert;
            integrator->_traj[curID].segment(idx*3,3) = vert.segment(0,3);

            depa_pos.emplace_back(mesh->_mesh->verts[idx]);
        }


        for(size_t i = 0;i < mesh->_farBindPoints.size();++i){
            size_t idx = mesh->_farBindPoints[i];
            Vec4d vert;
            vert << mesh->_mesh->verts[idx][0],mesh->_mesh->verts[idx][1],mesh->_mesh->verts[idx][2],1.0;
            vert = far_T->Mat * vert;
            integrator->_traj[curID].segment(idx*3,3) = vert.segment(0,3);

            depa_pos.emplace_back(mesh->_mesh->verts[idx]);
        }

        set_output("depa",std::move(depa));
        size_t iter_idx = 0;

        VecXd deriv(mesh->_mesh->size() * 3);
        VecXd ruc(mesh->_freeDoFs.size()),dpuc(mesh->_freeDoFs.size()),dp(mesh->_mesh->size() * 3);
    
        _HValueBuffer.resize(mesh->_connMatrix.nonZeros());
        _HucValueBuffer.resize(mesh->_freeConnMatrix.nonZeros());

        const size_t max_iters = 20;
        const size_t max_linesearch = 20;
        _wolfeBuffer.resize(max_linesearch);

        size_t search_idx = 0;

        do{
            FEM_Scaler e0,e1,eg0;
            e0 = EvalObjDerivHessian(mesh,force_model,damping_model,integrator,deriv,_HValueBuffer);

            // {
            //     std::cout << "check derivative" << std::endl;
            //     size_t nm_dofs = integrator->_traj[curID].size();
            //     MatXd H_fd = MatXd(nm_dofs,nm_dofs);
            //     VecXd deriv_fd = deriv,deriv_tmp = deriv;
            //     FEM_Scaler e0_tmp;
            //     VecXd cur_frame_copy = integrator->_traj[curID];

            //     for(size_t i = 0;i < nm_dofs;++i){
            //         integrator->_traj[curID] = cur_frame_copy;
            //         FEM_Scaler step = cur_frame_copy[i] * 1e-8;
            //         step = fabs(step) < 1e-8 ? 1e-8 : step;

            //         integrator->_traj[curID][i] += step;
            //         e0_tmp = EvalObjDeriv(mesh,force_model,integrator,deriv_tmp);

            //         deriv_fd[i] = (e0_tmp - e0) / step;
            //         H_fd.col(i) = (deriv_tmp - deriv) / step;


            //     }
            //     integrator->_traj[curID] = cur_frame_copy;

            //     FEM_Scaler deriv_error = (deriv_fd - deriv).norm();
            //     FEM_Scaler H_error = (mesh->MapHMatrix(_HValueBuffer.data()).toDense() - H_fd).norm();

            //     if(deriv_error > 1e-4){
            //         std::cout << "D_ERROR : " << deriv_error << std::endl;
            //         std::cout << "deriv_norm : " << deriv_fd.norm() << "\t" << deriv.norm() << std::endl;
            //         // for(size_t i = 0;i < deriv_fd.size();++i){
            //         //     std::cout << "idx : " << i << "\t" << deriv_fd[i] << "\t" << deriv[i] << std::endl;
            //         // }
            //         // for(size_t i = 0;i < nm_dofs;++i)
            //         //     std::cout << "idx : " << i << "\t" << deriv[i] << "\t" << deriv_fd[i] << std::endl;
            //         // throw std::runtime_error("DERROR");
            //     }

            //     if(H_error > 1e-3){
            //         std::cout << "H_error : " << H_error << std::endl;
            //         // std::cout << std::setprecision(6) << "H_cmp : " << std::endl << mesh->MapHMatrix(_HValueBuffer.data()).toDense() << std::endl;
            //         // std::cout << std::setprecision(6) << "H_fd : " << std::endl << H_fd << std::endl;

            //         throw std::runtime_error("HERROR");
            //     }
            // }

            // throw std::runtime_error("INT_ERROR");

            MatHelper::RetrieveDoFs(deriv.data(),ruc.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());
            MatHelper::RetrieveDoFs(_HValueBuffer.data(),_HucValueBuffer.data(),mesh->_SpMatFreeDoFs.size(),mesh->_SpMatFreeDoFs.data());

            if(ruc.norm() < epsilon){
                std::cout << "[" << iter_idx << "]break with ruc = " << ruc.norm() << "\t < \t" << epsilon << std::endl;
                break;
            }
            ruc *= -1;

            _LUSolver.compute(mesh->MapHucMatrix(_HucValueBuffer.data()));
            dpuc = _LUSolver.solve(ruc);


            eg0 = -dpuc.dot(ruc);

            if(eg0 > 0){
                std::cerr << "non-negative descent direction detected " << eg0 << std::endl;
                throw std::runtime_error("non-negative descent direction");
            }
            dp.setZero();
            MatHelper::UpdateDoFs(dpuc.data(),dp.data(),mesh->_freeDoFs.size(),mesh->_freeDoFs.data());


            search_idx = 0;

            // if(ruc.norm() < epsilon || \
            //         fabs(eg0) < epsilon * epsilon){

            //     std::cout << "break with ruc = " << ruc.norm() << "\t <  \t" << epsilon  << "\t or \t" << eg0 << "\t < \t" << epsilon * epsilon << std::endl;
            //     break;
            // }
            FEM_Scaler alpha = 2.0f;
            FEM_Scaler beta = 0.5f;
            FEM_Scaler c1 = 0.01f;

            // get_state_ref(0) += dp;
            double armijo_condition;

            do{
                if(search_idx != 0)
                    integrator->_traj[curID] -= alpha * dp;
                alpha *= beta;
                integrator->_traj[curID] += alpha * dp;
                e1 = EvalObj(mesh,force_model,damping_model,integrator);
                ++search_idx;
                _wolfeBuffer[search_idx-1](0) = (e1 - e0)/alpha;
                _wolfeBuffer[search_idx-1](1) = eg0;

                armijo_condition = double(e1) - double(e0) - double(c1)*double(alpha)*double(eg0);
            }while(/*(e1 > e0 + c1*alpha*eg0)*/ armijo_condition > 0.0f /* || (fabs(eg1) > c2*fabs(eg0))*/ && (search_idx < max_linesearch));

            if(search_idx == max_linesearch){
                std::cout << "LINESEARCH EXCEED" << std::endl;
                for(size_t i = 0;i < max_linesearch;++i)
                    std::cout << "idx:" << i << "\t" << _wolfeBuffer[i].transpose() << std::endl;
                throw std::runtime_error("LINESEARCH");
            }

            ++iter_idx;
        }while(iter_idx < max_iters);


        if(iter_idx == max_iters){
            std::cout << "MAX NEWTON ITERS EXCEED" << std::endl;
        }


        integrator->_stepID++;

        const VecXd& cur_frame = integrator->_traj[curID];
        auto resGeo = std::make_shared<PrimitiveObject>();
        auto &pos = resGeo->add_attr<zeno::vec3f>("pos");
        // std::cout << "OUTPUT FRAME " << cur_frame.norm() << std::endl;
        for(size_t i = 0;i < mesh->_mesh->size();++i){
            auto vert = cur_frame.segment(i*3,3);
            pos.emplace_back(vert[0],vert[1],vert[2]);
        }

        for(int i=0;i < mesh->_mesh->quads.size();++i){
            auto tet = mesh->_mesh->quads[i];
            resGeo->tris.emplace_back(tet[0],tet[1],tet[2]);
            resGeo->tris.emplace_back(tet[1],tet[3],tet[2]);
            resGeo->tris.emplace_back(tet[0],tet[2],tet[3]);
            resGeo->tris.emplace_back(tet[0],tet[3],tet[1]);
        }

        // std::cout << "POS : " << std::endl;
        // for(size_t i = 0;i < pos.size();++i){
        //     std::cout << "P<" << i << "> : \t" << pos[i][0] << "\t" << pos[i][1] << "\t" << pos[i][2] << std::endl;
        // }

        // std::cout << "size_of_frame : " << resGeo->tris.size() << "\t" << pos.size() << "\t" << cur_frame.size() <<  std::endl;
        // throw std::runtime_error("test frame");

        pos.resize(mesh->_mesh->size());
        set_output("curentFrame", resGeo);
    }

    FEM_Scaler EvalObj(const std::shared_ptr<FEMMesh>& mesh,
        const std::shared_ptr<MuscleForceModel>& muscle,
        const std::shared_ptr<DampingForceModel>& damp,
        const std::shared_ptr<FEMIntegrator>& integrator) {
            FEM_Scaler obj = 0;

            size_t clen = integrator->_intPtr->GetCouplingLength();
            size_t nm_elms = mesh->_mesh->quads.size();

            _objBuffer.resize(nm_elms);
            
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh);
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];
                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (integrator->_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],integrator->_traj[frameID]);
                }

                FEM_Scaler elm_obj = 0;
                integrator->_intPtr->EvalElmObj(attrbs,
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
        const std::shared_ptr<MuscleForceModel>& muscle,
        const std::shared_ptr<DampingForceModel>& damp,
        const std::shared_ptr<FEMIntegrator>& integrator,
        VecXd& deriv) {
            FEM_Scaler obj = 0;

            size_t clen = integrator->_intPtr->GetCouplingLength();
            size_t nm_elms = mesh->_mesh->quads.size();

            _objBuffer.resize(nm_elms);
            _derivBuffer.resize(nm_elms);
            
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh);
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];
                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (integrator->_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],integrator->_traj[frameID]);
                }

                FEM_Scaler elm_obj = 0;
                integrator->_intPtr->EvalElmObjDeriv(attrbs,
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
        const std::shared_ptr<MuscleForceModel>& muscle,
        const std::shared_ptr<DampingForceModel>& damp,
        const std::shared_ptr<FEMIntegrator>& integrator,
        VecXd& deriv,
        VecXd& HValBuffer) {
            FEM_Scaler obj = 0;
            size_t clen = integrator->_intPtr->GetCouplingLength();
            size_t nm_elms = mesh->_mesh->quads.size();

            _objBuffer.resize(nm_elms);
            _derivBuffer.resize(nm_elms);
            _HBuffer.resize(nm_elms);
            
            for(size_t elm_id = 0;elm_id < nm_elms;++elm_id){
                TetAttributes attrbs;
                AssignElmAttribs(elm_id,attrbs,mesh);
                // std::cout << "attrbs.elm_Act" << std::endl << attrbs._activation << std::endl;
                // throw std::runtime_error("CHECK ATTRS");
                std::vector<Vec12d> elm_traj(clen);
                auto tet = mesh->_mesh->quads[elm_id];
                for(size_t i = 0;i < clen;++i){
                    size_t frameID = (integrator->_stepID + clen - i) % clen;
                    RetrieveElmVector(tet,elm_traj[clen - i - 1],integrator->_traj[frameID]);
                }

                integrator->_intPtr->EvalElmObjDerivJacobi(attrbs,
                    muscle->_forceModel,
                    damp->_dampForce,
                    elm_traj,
                    &_objBuffer[elm_id],_derivBuffer[elm_id],_HBuffer[elm_id],true);


                // {
                //     FEM_Scaler obj_tmp;
                //     Vec12d deriv_tmp,deriv_fd;
                //     Mat12x12d H_fd;
                //     Vec12d frame_copy = elm_traj[clen - 1];

                //     for(size_t i = 0;i < 12;++i){
                //         elm_traj[clen - 1] = frame_copy;
                //         FEM_Scaler step = frame_copy[i] * 1e-8;
                //         step = fabs(step) < 1e-8 ? 1e-8 : step;
                //         elm_traj[clen - 1][i] += step;

                //         integrator->_intPtr->EvalElmObjDeriv(attrbs,
                //             muscle->_forceModel,
                //             damp->_dampForce,
                //             elm_traj,&obj_tmp,deriv_tmp);
                //         deriv_fd[i] = (obj_tmp - _objBuffer[elm_id])  / step;
                //         H_fd.col(i) = (deriv_tmp - _derivBuffer[elm_id]) / step;
                //     }

                //     elm_traj[clen - 1] = frame_copy;

                //     FEM_Scaler D_error = (deriv_fd - _derivBuffer[elm_id]).norm() / deriv_fd.norm();
                //     FEM_Scaler H_error = (H_fd - _HBuffer[elm_id]).norm()/H_fd.norm();

                //     // if(D_error > 1e-3){
                //     //     std::cout << "ELM_ID : " << elm_id << std::endl;
                //     //     std::cout << "INT_ELM_D_error : " << D_error << std::endl;
                //     //     for(size_t i = 0;i < 12;++i)
                //     //         std::cout << "idx : " << i << "\t" << deriv_fd[i] << "\t" << _derivBuffer[elm_id][i] << std::endl;
                //     //     throw std::runtime_error("INT_ELM_D_ERROR");
                //     // }

                //     if(H_error > 1e-3){
                //         std::cout << "ELM_ID : " << elm_id << std::endl;
                //         std::cout << "D_Error : " << D_error << std::endl;
                //         std::cout << "INT_ELM_H_Error : " << H_error << std::endl;
                //         std::cout << "H_cmp : " << std::endl << _HBuffer[elm_id] << std::endl;
                //         std::cout << "H_fd : " << std::endl << H_fd << std::endl;
                //         throw std::runtime_error("INT_ELM_H_ERROR");
                //     }
                //     // throw std::runtime_error("INT_ERROR_CHECK");
                // }


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

    void AssignElmAttribs(size_t elm_id,TetAttributes& attrbs,const std::shared_ptr<FEMMesh>& mesh) const {
        attrbs._elmID = elm_id;
        attrbs._Minv = mesh->_elmMinv[elm_id];
        attrbs._dFdX = mesh->_elmdFdx[elm_id];
        attrbs._fiberOrient = mesh->_elmOrient[elm_id];
        attrbs._fiberWeight = mesh->_elmWeight[elm_id];
        attrbs._activation = mesh->_elmAct[elm_id];
        attrbs._E = mesh->_elmYoungModulus[elm_id];
        attrbs._nu = mesh->_elmPossonRatio[elm_id];
        attrbs._d = mesh->_elmDamp[elm_id];
        attrbs._volume = mesh->_elmVolume[elm_id];
        attrbs._density = mesh->_elmDensity[elm_id];
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

    VecXd _HValueBuffer;
    VecXd _HucValueBuffer;
    VecXd _FreeJacobiBuffer;
    std::vector<FEM_Scaler> _objBuffer;
    std::vector<Vec12d> _derivBuffer;
    std::vector<Mat12x12d> _HBuffer;

    Eigen::SparseLU<SpMat> _LUSolver;

    std::vector<Vec2d> _wolfeBuffer;
};

ZENDEFNODE(DoTimeStep,{
    {{"mesh"},{"muscleForce"},{"dampForce"},{"integrator"},{"epsilon"},{"CT"},{"FT"}},
    {"curentFrame","depa"},
    {},
    {"FEM"},
});

};