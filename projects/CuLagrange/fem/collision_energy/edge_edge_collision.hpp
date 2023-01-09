#pragma once

#include "collision_utils.hpp"

namespace zeno {
namespace EDGE_EDGE_COLLISION {
    using namespace COLLISION_UTILS;

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr REAL psi(const VECTOR3 v[4],
                            const VECTOR2& a, 
                            const VECTOR2& b,
                            const REAL& _mu,
                            const REAL& _nu,
                            const REAL& _eps){
        // convert to vertices and edges
        VECTOR3 e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];

        // get the normal
        VECTOR3 n = e[1].cross(e[0]);

        // if the two are co-linear, skip the normalization 
        if (n.norm() > (REAL)1e-8)
            n = n / n.norm();

        // get the interpolated vertices
        const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
        const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
        const VECTOR3 diff = vb - va;
        
        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;

        // get the spring length, non-zero rest-length
        const REAL springLength = _eps + diff.dot(sign * n);
        return _mu * springLength * springLength;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr REAL psiNegated(const VECTOR3 v[4],
                                const VECTOR2& a, 
                                const VECTOR2& b,
                                const REAL& _mu,
                                const REAL& _nu,
                                const REAL& _eps)
    {
        // convert to vertices and edges
        VECTOR3 e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];
        // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
        // will pick up the slack, so ignore it. But ... the collision is still well
        // defined from the perspective of the pinned version of this energy.
        //
        // Regardless, the conditioning of the force goes crazy in the near-parallel case,
        // so we should skip it here.
        if (nearlyParallel(e))
            return (REAL)0.0;

        // get the normal
        VECTOR3 n = e[1].cross(e[0]);

        // if the two are co-linear, skip the normalization (this may be redundant with
        // the nearlyParallel check)
        if (n.norm() > (REAL)1e-8)
            n = n / n.norm();

        // get the interpolated vertices
        const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
        const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
        const VECTOR3 diff = vb - va;
        
        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;

        // get the spring length, non-zero rest-length
        const REAL springLength = _eps - diff.dot(sign * n);
        return _mu * springLength * springLength;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR12 gradient(const VECTOR3 v[4],
                                const VECTOR2& a, 
                                const VECTOR2& b,
                                const REAL& _mu,
                                const REAL& _nu,
                                const REAL& _eps)
    {
        // convert to vertices and edges
        VECTOR3 e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];
        // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
        // will pick up the slack, so ignore it. But ... the collision is still well
        // defined from the perspective of the pinned version of this energy.
        //
        // Regardless, the conditioning of the force goes crazy in the near-parallel case,
        // so we should skip it here.
        if (nearlyParallel(e))
            return VECTOR12::zeros();

        // get the normal
        VECTOR3 n = e[1].cross(e[0]);
        
        // if the two are co-linear, skip the normalization 
        if (n.norm() > (REAL)1e-8)
            n = n / n.norm();
        
        // get the interpolated vertices
        const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
        const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
        const VECTOR3 diff = vb - va;

        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;
        const REAL springLength = _eps + diff.dot(sign * n);
        return (REAL)2.0 * _mu * springLength * springLengthGradient(e,n,diff,a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR12 gradientNegated(const VECTOR3 v[4],
                                const VECTOR2& a, 
                                const VECTOR2& b,
                                const REAL& _mu,
                                const REAL& _nu,
                                const REAL& _eps)
    {
        // convert to vertices and edges
        VECTOR3 e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];

        // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
        // will pick up the slack, so ignore it. But ... the collision is still well
        // defined from the perspective of the pinned version of this energy.
        //
        // Regardless, the conditioning of the force goes crazy in the near-parallel case,
        // so we should skip it here.
        if (nearlyParallel(e))
            return VECTOR12::zeros();

        // get the normal
        VECTOR3 n = e[1].cross(e[0]);
        
        // if the two are co-linear, skip the normalization (this may be redundant with
        // the nearlyParallel check)
        if (n.norm() > (REAL)1e-8)
            n = n / n.norm();
        
        // get the interpolated vertices
        const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
        const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
        const VECTOR3 diff = vb - va;

        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;

        //const REAL springLength = _eps + diff.dot(sign * n);
        //return 2.0 * _mu * springLength * springLengthGradient(e,n,diff,a,b);
        const REAL springLength = _eps - diff.dot(sign * n);
        return (REAL)-2.0 * _mu * springLength * springLengthGradient(e,n,diff,a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    // partial of (va - vb)
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX3x12 vDiffPartial(const VECTOR2& a, const VECTOR2& b)
    {
        MATRIX3x12 tPartial{(REAL)0.0};
        // tPartial.setZero();
        tPartial(0,0) = tPartial(1,1)  = tPartial(2,2) = -a[0];
        tPartial(0,3) = tPartial(1,4)  = tPartial(2,5) = -a[1];
        tPartial(0,6) = tPartial(1,7)  = tPartial(2,8) = b[0];
        tPartial(0,9) = tPartial(1,10) = tPartial(2,11) = b[1];

        return tPartial;
    }


    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 hessian(const VECTOR3 v[4],
                                const VECTOR2& a, 
                                const VECTOR2& b,
                                const REAL& _mu,
                                const REAL& _nu,
                                const REAL& _eps){
        // convert to vertices and edges
        VECTOR3 e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];
        
        // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
        // will pick up the slack, so ignore it. But ... the collision is still well
        // defined from the perspective of the pinned version of this energy.
        //
        // Regardless, the conditioning of the force goes crazy in the near-parallel case,
        // so we should skip it here.
        if (nearlyParallel(e))
            return MATRIX12::zeros();

        // get the normal
        VECTOR3 n = e[1].cross(e[0]);
        
        // if the two are co-linear, skip the normalization 
        if (n.norm() > (REAL)1e-8)
            n = n / n.norm();

        // get the interpolated vertices
        const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
        const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
        const VECTOR3 diff = vb - va;

        //if (diff.dot(n) > 0.0)
        //  n *= -1.0;
        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;

        // get the spring length, non-zero rest-length
        const REAL springLength = _eps + diff.dot(sign * n);

        // ndotGrad    = ndot_gradient(x);
        const VECTOR12 springLengthGrad = springLengthGradient(e,n,diff,a,b);

        // ndotHessian = ndot_hessian(x);
        const MATRIX12 springLengthH = springLengthHessian(e,n,diff,a,b);
        
        return (REAL)2.0 * _mu * (dyadic_prod(springLengthGrad,springLengthGrad) +
                            springLength * springLengthH);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 hessianNegated(const VECTOR3 v[4],
                                const VECTOR2& a, 
                                const VECTOR2& b,
                                const REAL& _mu,
                                const REAL& _nu,
                                const REAL& _eps)
    {
        // convert to vertices and edges
        VECTOR3 e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2];
        
        // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
        // will pick up the slack, so ignore it. But ... the collision is still well
        // defined from the perspective of the pinned version of this energy.
        //
        // Regardless, the conditioning of the force goes crazy in the near-parallel case,
        // so we should skip it here.
        if (nearlyParallel(e))
            return MATRIX12::zeros();

        // get the normal
        VECTOR3 n = e[1].cross(e[0]);
        
        // if the two are co-linear, skip the normalization (this may be redundant with
        // the nearlyParallel check)
        if (n.norm() > 1e-8)
            n = n / n.norm();

        // get the interpolated vertices
        const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
        const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
        const VECTOR3 diff = vb - va;

        //if (diff.dot(n) > 0.0)
        //  n *= -1.0;
        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;

        // get the spring length, non-zero rest-length
        //const REAL springLength = _eps + diff.dot(sign * n);
        const REAL springLength = _eps - diff.dot(sign * n);

        // ndotGrad    = ndot_gradient(x);
        const VECTOR12 springLengthGrad = springLengthGradient(e,n,diff,a,b);

        // ndotHessian = ndot_hessian(x);
        const MATRIX12 springLengthH = springLengthHessian(e,n,diff,a,b);
        
        //return 2.0 * _mu * (springLengthGrad * springLengthGrad.transpose() +
        //                    springLength * springLengthH);
        return (REAL)-2.0 * _mu * (springLength * springLengthH - 
                            zs::dyadic_prod(springLengthGrad,springLengthGrad));
    }

};
};