#pragma once

#include "collision_utils.hpp"

// using namespace std;
namespace zeno {

namespace VERTEX_FACE_SQRT_COLLISION {

    #define _inverseEps 1e-4

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

        const bool reversal = !reverse(v,e);

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
//  our normal pointing outward
    constexpr VECTOR12 gradient(const VECTOR3 v[4], const VECTOR3& bary,const REAL& _mu,const REAL& _nu,const REAL& _eps)
    {
        // REAL _inverseEps = 1e-6;
        using DREAL = double;
        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 
        const bool reversal = !reverse(v,e);
        
        // remember we had to reorder vertices in a wonky way
        const VECTOR3 xs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
        const VECTOR3 t = v[0] - xs;
        // const REAL tMagnitude = t.norm();
        const DREAL tMagnitude = t.template cast<DREAL>().norm();
        //const REAL springDiff = tMagnitude - _eps;
        const DREAL springDiff = (reversal) ? tMagnitude + _eps : tMagnitude - _eps;
        const MATRIX3x12 tDiff = tDiffPartial(bary); 

        // printf("springDiff : %f\n t : %f %f %f\n",
        //     (float)springDiff,(float)t[0],(float)t[1],(float)t[2]);

        // if(reversal && tMagnitude > _eps)
        //     printf("invalid direction dectected on reversal");


        // if everything has become undefined, just give up
        // const REAL tDott = t.dot(t);
        // if (tMagnitude <= _inverseEps || zs::abs(tDott) < _inverseEps)
        //     return VECTOR12::zeros();

        if (tMagnitude <= _inverseEps)
            return VECTOR12::zeros();

        // const DREAL alpha = (DREAL)springDiff/tMagnitude;
        const auto tn = t.template cast<DREAL>().normalized().template cast<REAL>();

        // const VECTOR12 result = (REAL)2.0 * _mu * springDiff * ((REAL)1.0 / tMagnitude) * tDiff.transpose() * t;

        VECTOR12 result = (REAL)2.0 * _mu * (REAL)springDiff * tDiff.transpose() * tn;

        // could instead try to trap all the inverses and hand back something fixed up,
        // but consistency is not guaranteed, so let's just zero it out at the first
        // sign of trouble
        //const REAL tMagnitudeInv = (zs::abs(tMagnitude) > _inverseEps) ? 1.0 / tMagnitude : 0.0;
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
        // REAL _inverseEps = 1e-6;

        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 
        const bool reversal = !reverse(v,e);
        
        using DREAL = double;

        // remember we had to reorder vertices in a wonky way
        const VECTOR3 xs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
        const VECTOR3 t = v[0] - xs;
        // const DREAL tDott = t.dot(t);
        // const REAL tMagnitude = zs::sqrt(tDott);
        const DREAL tMagnitude = t.cast<DREAL>().norm();
        //const REAL springDiff = tMagnitude - _eps;
        const DREAL springDiff = (reversal) ? tMagnitude + _eps : tMagnitude - _eps;
        const MATRIX3x12 tDiff = tDiffPartial(bary); 

        // get the spring length, non-zero rest-length
        const VECTOR12 product = tDiff.transpose() * t;

        // if everything has become undefined, just give up
        if (tMagnitude <= _inverseEps)
            return MATRIX12::zeros();

        // auto alpha = ((REAL)1.0 / tMagnitude / tMagnitude - springDiff / (tMagnitude * tMagnitude * tMagnitude));
        auto alpha = 1.0 - springDiff/tMagnitude;
        auto beta = (springDiff / tMagnitude);

        auto tn = t.template cast<DREAL>().normalized().template cast<REAL>();
        auto productn = tDiff.transpose() * tn;

        alpha = alpha > 0 ? alpha : 0;
        beta = beta > 0 ? beta : 0;

        // return (REAL)2.0 * _mu * (((REAL)1.0 / tDott - springDiff / (tDott * tMagnitude)) * (zs::dyadic_prod(product,product)) +
        //                     (springDiff / tMagnitude) * tDiff.transpose() * tDiff); 

        // return (REAL)2.0 * _mu * (alpha * (zs::dyadic_prod(product,product)) + beta * tDiff.transpose() * tDiff); 

        return (REAL)2.0 * _mu * ((REAL)alpha * (zs::dyadic_prod(productn,productn)) + (REAL)beta * tDiff.transpose() * tDiff);

        // could instead try to trap all the inverses and hand back something fixed up,
        // but consistency is not guaranteed, so let's just zero it out at the first
        // sign of trouble
        //const REAL tMagnitudeInv = (zs::abs(tMagnitude) > _inverseEps) ? 1.0 / tMagnitude : 0.0;
        //const REAL tDottInv = (zs::abs(tDott) > _inverseEps) ? 1.0 / tDott : 1.0;
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