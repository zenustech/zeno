#pragma once

#include "collision_utils.hpp"

// #include "zensim/cuda/execution/ExecutionPolicy.cuh"
// #include "zensim/omp/execution/ExecutionPolicy.hpp"
// #include "zensim/container/Bvh.hpp

namespace zeno {
namespace VERTEX_FACE_COLLISION {
    using namespace COLLISION_UTILS;

    REAL psi(const std::vector<VECTOR3>& v,const REAL& mu,const REAL& nu,const REAL& eps) {        
        std::vector<VECTOR3> e{3};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];

        // auto n = zs::cross(e[2],e[0]);
        auto n = e[2].cross(e[0]);
        n = n / n.norm();

        auto tvf = v[0] - v[2];
        auto springLength = n.dot(tvf) - eps;

        return mu * springLength * springLength;
    }

    REAL psi(VECTOR12& x,const REAL& mu,const REAL& nu,const REAL& eps) {
        std::vector<VECTOR3> v(4);
        for(int i = 0;i < 4;++i)
            for(int j = 0;j < 3;++j)
                v[i][j] = x[i * 3 + j];
        return psi(v,mu,nu,eps);        
    }

    VECTOR12 gradient(const std::vector<zs::vec<REAL,3>>& v,const REAL& mu,const REAL& nu,const REAL& eps) {
        std::vector<zs::vec<REAL,3>> e;

        e.resize(3);
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];      

        // get the normal
        VECTOR3 n = e[2].cross(e[0]);
        n = n / n.norm();

        const auto& tvf= e[1];
        REAL springLength = tvf.dot(n) - eps;  

        return (REAL)2.0 * mu * springLength * springLengthGradient(v,e,n);
    }

    auto gradient(VECTOR12& x,const REAL& mu,const REAL& nu,const REAL& eps) {
        std::vector<VECTOR3> v(4);
        for(int i = 0;i < 4;++i)
            for(int j = 0;j < 3;++j)
                v[i][j] = x[i * 3 + j];
        return gradient(v,mu,nu,eps);             
    }

    // MATRIX12 dyadic(const VECTOR12& v1,const VECTOR12& v2) {
    //     auto res = MATRIX12::zeros();
    //     for(int i = 0;i < 12;++i)
    //         for(int j = 0;j < 12;++j)
    //             res(i,j) = v1[i] * v2[j];
    // }

    auto hessian(const std::vector<zs::vec<REAL,3>>& v,const REAL& mu,const REAL& nu,const REAL& eps) {
        std::vector<zs::vec<REAL,3>> e;

        e.resize(3);
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 

        // get the normal
        VECTOR3 n = e[2].cross(e[0]);
        n = n / n.norm();

        // get the spring length, non-zero rest-length
        const VECTOR3 tvf = v[0] - v[2];
        const REAL springLength = n.dot(tvf) - eps;

        // ndotGrad    = ndot_gradient(x);
        const VECTOR12 gvf = springLengthGradient(v,e,n);

        // ndotHessian = ndot_hessian(x);
        const MATRIX12 springLengthH = springLengthHessian(v,e,n);
        
        // final = 2 * k * (ndotGrad * ndotGrad' + ndot * ndotHessian);
        return (REAL)2.0 * mu * (zs::dyadic_prod(gvf, gvf) + 
                            springLength * springLengthH);
    }

    // auto hessian(const VECTOR12& x,const REAL& mu,const REAL& nu) {
    //     return zs::vec<REAL,12,12>::zeros();
    // }

};

};