#pragma once

#include "collision_utils.hpp"

// using namespace std;
namespace zeno {

namespace VERTEX_FACE_SQRT_COLLISION {

    using namespace COLLISION_UTILS;
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// TODO: for the purpose of consistent energy interface, we use vec4 bary, with bary = vec4{1,tri_bary}
    constexpr REAL psi(const VECTOR3 v[4],const VECTOR3& bary,const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 

        const bool reversal = reverse(v,e);

        const VECTOR3 xs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
        const VECTOR3 t = v[0] - xs;
        const REAL tMagnitude = t.norm();
        const REAL springDiff = (reversal) ? tMagnitude + _eps : tMagnitude - _eps;

        return _mu * springDiff * springDiff;
    }    

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr REAL psi(const VECTOR3 v[4],const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        const VECTOR3 bary = getInsideBarycentricCoordinates(v);
        return psi(v,bary,_mu,_nu,_eps);
    }

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
    constexpr VECTOR12 gradient(const VECTOR3 v[4], const VECTOR3& bary,const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        REAL _inverseEps = 1e-8;

        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 
        const bool reversal = reverse(v,e);
        
        // remember we had to reorder vertices in a wonky way
        const VECTOR3 xs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
        const VECTOR3 t = v[0] - xs;
        const REAL tMagnitude = t.norm();
        //const REAL springDiff = tMagnitude - _eps;
        const REAL springDiff = (reversal) ? tMagnitude + _eps : tMagnitude - _eps;
        const MATRIX3x12 tDiff = tDiffPartial(bary); 

        // if everything has become undefined, just give up
        const REAL tDott = t.dot(t);
        if (fabs(tMagnitude) <= _inverseEps || fabs(tDott) < _inverseEps)
            return VECTOR12::zeros();

        const VECTOR12 result = (REAL)2.0 * _mu * springDiff * ((REAL)1.0 / tMagnitude) * tDiff.transpose() * t;

        // could instead try to trap all the inverses and hand back something fixed up,
        // but consistency is not guaranteed, so let's just zero it out at the first
        // sign of trouble
        //const REAL tMagnitudeInv = (fabs(tMagnitude) > _inverseEps) ? 1.0 / tMagnitude : 0.0;
        //const VECTOR12 result = 2.0 * _mu * springDiff * tMagnitudeInv * tDiff.transpose() * t

        return result;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR12 gradient(const VECTOR3 v[4],const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        const VECTOR3 bary = getInsideBarycentricCoordinates(v);
        return gradient(v, bary, _mu, _nu, _eps);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 hessian(const VECTOR3 v[4], const VECTOR3& bary,const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        REAL _inverseEps = 1e-8;

        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 
        const bool reversal = reverse(v,e);
        
        // remember we had to reorder vertices in a wonky way
        const VECTOR3 xs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
        const VECTOR3 t = v[0] - xs;
        const REAL tDott = t.dot(t);
        const REAL tMagnitude = zs::sqrt(tDott);
        //const REAL springDiff = tMagnitude - _eps;
        const REAL springDiff = (reversal) ? tMagnitude + _eps : tMagnitude - _eps;
        const MATRIX3x12 tDiff = tDiffPartial(bary); 

        // get the spring length, non-zero rest-length
        const VECTOR12 product = tDiff.transpose() * t;

        // if everything has become undefined, just give up
        if (fabs(tMagnitude) <= _inverseEps || fabs(tDott) < _inverseEps)
            return MATRIX12::zeros();

        return (REAL)2.0 * _mu * (((REAL)1.0 / tDott - springDiff / (tDott * tMagnitude)) * (zs::dyadic_prod(product,product)) +
                            (springDiff / tMagnitude) * tDiff.transpose() * tDiff); 

        // could instead try to trap all the inverses and hand back something fixed up,
        // but consistency is not guaranteed, so let's just zero it out at the first
        // sign of trouble
        //const REAL tMagnitudeInv = (fabs(tMagnitude) > _inverseEps) ? 1.0 / tMagnitude : 0.0;
        //const REAL tDottInv = (fabs(tDott) > _inverseEps) ? 1.0 / tDott : 1.0;
        //return 2.0 * _mu * ((tDottInv - springDiff / (tDott * tMagnitude)) * (product * product.transpose()) +
        //                    (springDiff * tMagnitudeInv) * tDiff.transpose() * tDiff); 
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 hessian(const VECTOR3 v[4],const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        const VECTOR3 bary = getInsideBarycentricCoordinates(v);
        return hessian(v, bary,_mu,_nu,_eps);
    }



};

};