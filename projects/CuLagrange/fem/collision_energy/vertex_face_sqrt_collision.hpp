#pragma once

#include "collision_utils.hpp"

// using namespace std;
namespace zeno {

namespace VERTEX_FACE_SQRT_COLLISION {

    #define _inverseEps 1e-6

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
        const VECTOR3 bary = LSL_GEO::getInsideBarycentricCoordinates(v);
        return psi(v,bary,_mu,_nu,_eps);
    }

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
//  our normal pointing outward
    constexpr VECTOR12 gradient(const VECTOR3 v[4], const VECTOR3& bary,const REAL& _mu,const REAL& _nu,const REAL& _eps,bool collide_from_inside = false)
    {
        // REAL _inverseEps = 1e-6;
        using DREAL = double;
        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 
        bool reversal = !reverse(v,e);
        if(collide_from_inside)
            reversal = !reversal;
        
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
        const REAL tDott = t.dot(t);
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

        if(zs::isnan(result.norm())) {
            printf("nan cH detected %f %f %f\n",(float)springDiff,(float)tDiff.norm(),(float)tn.norm());
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR12 gradient(const VECTOR3 v[4],const REAL& _mu,const REAL& _nu,const REAL& _eps,bool collide_from_inside = false)
    {
        const VECTOR3 bary = LSL_GEO::getInsideBarycentricCoordinates(v);
        return gradient(v, bary, _mu, _nu, _eps,collide_from_inside);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 hessian(const VECTOR3 v[4], const VECTOR3& bary,const REAL& _mu,const REAL& _nu,const REAL& _eps,bool collide_from_inside = false)
    {
        // REAL _inverseEps = 1e-6;

        // convert to vertices and edges
        zs::vec<REAL,3> e[3] = {};
        e[0] = v[3] - v[2];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[2]; 
        bool reversal = !reverse(v,e);
        if(collide_from_inside)
            reversal = !reversal;

#if 1

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
        const REAL tDott = t.dot(t);
        // if (tMagnitude <= _inverseEps || zs::abs(tDott) < _inverseEps)
        //     return MATRIX12::zeros();
        if (tMagnitude <= _inverseEps)
            return MATRIX12::zeros();

        // auto alpha = ((REAL)1.0 / tMagnitude / tMagnitude - springDiff / (tMagnitude * tMagnitude * tMagnitude));
        auto alpha = 1.0 - springDiff/tMagnitude;
        auto beta = (springDiff / tMagnitude);

        auto tn = t.template cast<DREAL>().normalized().template cast<REAL>();
        auto productn = tDiff.transpose() * tn;

        alpha = alpha > 0 ? alpha : 0;
        beta = beta > 0 ? beta : 0;

        auto result = (REAL)2.0 * _mu * ((REAL)alpha * (zs::dyadic_prod(productn,productn)) + (REAL)beta * tDiff.transpose() * tDiff);
        if(zs::isnan(result.norm())) {
            printf("nan cH detected %f %f %f %f\n",(float)alpha,(float)productn.norm(),(float)beta,(float)tDiff.norm());
        }

        return (REAL)2.0 * _mu * ((REAL)alpha * (zs::dyadic_prod(productn,productn)) + (REAL)beta * tDiff.transpose() * tDiff);
        // auto H = (REAL)2.0 * _mu * ((REAL)alpha * (zs::dyadic_prod(productn,productn)) + (REAL)beta * tDiff.transpose() * tDiff);
        // make_pd(H);
        // return H; 


        // return (REAL)2.0 * _mu * (((REAL)1.0 / tDott - springDiff / (tDott * tMagnitude)) * (zs::dyadic_prod(product,product)) +
        //                     (springDiff / tMagnitude) * tDiff.transpose() * tDiff); 

        // return (REAL)2.0 * _mu * (alpha * (zs::dyadic_prod(product,product)) + beta * tDiff.transpose() * tDiff); 

        // could instead try to trap all the inverses and hand back something fixed up,
        // but consistency is not guaranteed, so let's just zero it out at the first
        // sign of trouble
        //const REAL tMagnitudeInv = (zs::abs(tMagnitude) > _inverseEps) ? 1.0 / tMagnitude : 0.0;
        //const REAL tDottInv = (zs::abs(tDott) > _inverseEps) ? 1.0 / tDott : 1.0;
        //return 2.0 * _mu * ((tDottInv - springDiff / (tDott * tMagnitude)) * (product * product.transpose()) +
        //                    (springDiff * tMagnitudeInv) * tDiff.transpose() * tDiff); 

#else
        const VECTOR3 xs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
        const VECTOR3 t = v[0] - xs;  

        const REAL tDott = t.dot(t);
        const REAL tMagnitude = zs::sqrt(tDott);


        const REAL springDiff = (reversal) ? tMagnitude + _eps : tMagnitude - _eps;
        const MATRIX3x12 tDiff = tDiffPartial(bary); 

        // get the spring length, non-zero rest-length
        const VECTOR12 product = tDiff.transpose() * t;

        auto res =  (REAL)2.0 * _mu * (((REAL)1.0 / tDott - springDiff / (tDott * tMagnitude)) * (zs::dyadic_prod(product,product)) + (springDiff / tMagnitude) * tDiff.transpose() * tDiff);   
        make_pd(res);
        return res;    

#endif
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 hessian(const VECTOR3 v[4],const REAL& _mu,const REAL& _nu,const REAL& _eps,bool collide_from_inside = false)
    {
        const VECTOR3 bary = LSL_GEO::getInsideBarycentricCoordinates(v);
        return hessian(v, bary,_mu,_nu,_eps,collide_from_inside);
    }

    // constexpr VECTOR12 damp_gradient(const VECTOR v[4],const VECTOR vp[4],const REAL& _dt, const VECTOR3& bary,const REAL& _kd,const REAL& _mu,const REAL& _nu,const REAL& eps){
    //     using DREAL = double;

    //     // const VECTOR3 vs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
    //     const VECTOR3 t = v[0] - (bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3]);// relative position
    //     // const VECTOR3 vps = bary[0] * vp[1] + bary[1] * vp[2] + bary[2] * vp[3];
    //     const VECTOR3 tp = vp[0] - (bary[0] * vp[1] + bary[1] * vp[2] + bary[2] * vp[3]);// previous relative position
    //     const VECTOR3 vel_t = (t - tp) / _dt;// relative velocity

    //     const MATRIX3x12 tDiff = tDiffPartial(bary); 
    //     const auto tn = t.template cast<DREAL>().normalized().template cast<REAL>();

    //     const DREAL project_vel_t = vel_t.dot(tn);
    //     return (REAL)2.0 * _mu * _kd * (REAL)project_vel_t * tDiff.transpose() * tn;
    // }

    // constexpr VECTOR12 damp_gradient(const VECTOR v[4],const VECTOR vp[4],const REAL& _dt,const REAL& _kd,const REAL& _mu,const REAL& _nu,const REAL& eps)
    // {
    //     const VECTOR3 bary = LSL_GEO::getInsideBarycentricCoordinates(v);
    //     return damp_gradient(v, vp,_dt,bary,_kd, _mu,_nu,eps);
    // }

    // const MATRIX12 damp_hessian(const VECTOR v[4],const VECTOR vp[4],const REAL& _dt, const VECTOR3& bary,const REAL& _kd,const REAL& _mu,const REAL& _nu,const REAL& eps) {
    //     using DREAL = double;

    //     // const VECTOR3 vs = bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3];
    //     const VECTOR3 t = v[0] - (bary[0] * v[1] + bary[1] * v[2] + bary[2] * v[3]);// relative position
    //     // const VECTOR3 vps = bary[0] * vp[1] + bary[1] * vp[2] + bary[2] * vp[3];
    //     const VECTOR3 tp = vp[0] - (bary[0] * vp[1] + bary[1] * vp[2] + bary[2] * vp[3]);// previous relative position
    //     const VECTOR3 vel_t = (t - tp) / _dt;// relative velocity

    //     const MATRIX3x12 tDiff = tDiffPartial(bary); 
    //     const REAL tDott = t.dot(t);
    //     const REAL tMagnitude = zs::sqrt(tDott);

    //     const VECTOR12 product = tDiff.transpose() * t;
    //     const VECTOR12 vproduct = tDiff.transpose() * vel_t;

    //     const DREAL project_vel_t = vel_t.dot(tn);

    //     return (T)2.0 * mu * kd * (
    //         ((T)1.0/_dt/tDott - (T)2.0*project_vel_t/tMagnitude/tDott)*zs::dyadic_prod(product,product) +
    //             project_vel_t/tDott * zs::dyadic_prod(tDiff) +
    //             (T)1.0/tDott * zs::dyadic_prod(product,product)
    //     );

    // }




};

};