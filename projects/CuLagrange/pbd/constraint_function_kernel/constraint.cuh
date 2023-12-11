#pragma once

#include "../../geometry/kernel/geo_math.hpp"
#include "zensim/math/MathUtils.h"
#include "zensim/math/DihedralAngle.hpp"

namespace zeno { namespace CONSTRAINT {
// FOR CLOTH SIMULATION
    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_DistanceConstraint(
        const VECTOR3d &p0, const SCALER& invMass0, 
        const VECTOR3d &p1, const SCALER& invMass1,
        const SCALER& restLength,
        const SCALER& xpbd_affliation,
        const SCALER& dt,
        SCALER& lambda,
        VECTOR3d &corr0, VECTOR3d &corr1)
    {				
        SCALER K = invMass0 + invMass1;
        if(K < static_cast<SCALER>(1e-6)) {
            // printf("abondoned stretch solve due to too small K : %f\n",(float)K);
            return false;
        }

        VECTOR3d n = p0 - p1;
        SCALER d = n.norm();
        SCALER C = d - restLength;

        if (d > static_cast<SCALER>(1e-6))
            n /= d;
        else
        {
            corr0 = VECTOR3d::uniform(0);
            corr1 = VECTOR3d::uniform(0);
            // printf("abondoned stretch solve due to too small d : %f\n",(float)d);
            return false;
        }

        SCALER alpha = 0.0;
        if (xpbd_affliation != 0.0)
        {
            alpha = static_cast<SCALER>(1.0) / (xpbd_affliation * dt * dt);
            K += alpha;
        }

        SCALER Kinv = 0.0;
        if (zs::abs(K) > static_cast<SCALER>(1e-6))
            Kinv = static_cast<SCALER>(1.0) / K;
        else
        {
            corr0 = VECTOR3d::uniform(0);
            corr1 = VECTOR3d::uniform(0);
            // printf("abondoned stretch solve due to too small K2 : %f\n",(float)K);
            return false;
        }



        const SCALER delta_lambda = -Kinv * (C + alpha * lambda);
        auto ori_lambda = lambda;
        auto al = alpha * lambda;

        lambda += delta_lambda;

        // printf("input l : %f and rl : %f delta_lambda : %f : alpha * lambda : %f ori_lambda = %f\n",
        //     (float)d,(float)restLength,(float)delta_lambda,(float)al,(float)ori_lambda);

        const VECTOR3d pt = n * delta_lambda;

        corr0 =  invMass0 * pt;
        corr1 = -invMass1 * pt;
        return true;
    }

    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_DistanceConstraint(
        const VECTOR3d &p0, const SCALER& invMass0, 
        const VECTOR3d &p1, const SCALER& invMass1,
        const VECTOR3d &pp0,
        const VECTOR3d &pp1,
        const SCALER& restLength,
        const SCALER& xpbd_affiliation,
        const SCALER& kdamp_ratio,
        const SCALER& dt,
        SCALER& lambda,
        VECTOR3d &corr0, VECTOR3d &corr1)
    {				
        SCALER wsum = invMass0 + invMass1;
        if(wsum < static_cast<SCALER>(1e-6)) {
            // printf("abondoned stretch solve due to too small K : %f\n",(float)K);
            return false;
        }

        VECTOR3d n = p0 - p1;
        SCALER d = n.norm();
        SCALER C = d - restLength;

        if (d > static_cast<SCALER>(1e-6))
            n /= d;
        else
        {
            corr0 = VECTOR3d::uniform(0);
            corr1 = VECTOR3d::uniform(0);
            // printf("abondoned stretch solve due to too small d : %f\n",(float)d);
            return false;
        }


        SCALER alpha = 0.0;
        if (xpbd_affiliation != 0.0)
        {
            alpha = static_cast<SCALER>(1.0) / (xpbd_affiliation * dt * dt);
        }

        const auto& gradC = n;

        SCALER dsum = 0.0, gamma = 1.0;
        if(kdamp_ratio > 0) {
            auto beta = kdamp_ratio * dt * dt;
            gamma = alpha * beta / dt;
            dsum = gamma * n.dot((p0 - pp0) - (p1 - pp1));
            // gamma += 1.0;
        }

        const SCALER delta_lambda = -(C + alpha * lambda + dsum) / ((gamma + static_cast<SCALER>(1.0)) * wsum + alpha);

        lambda += delta_lambda;

        const VECTOR3d pt = n * delta_lambda;

        corr0 =  invMass0 * pt;
        corr1 = -invMass1 * pt;
        return true;
    }


    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_DistanceConstraint(
        const VECTOR3d &p0, SCALER invMass0, 
        const VECTOR3d &p1, SCALER invMass1,
        const SCALER expectedDistance,
        const SCALER stiffness,
        VECTOR3d &corr0, VECTOR3d &corr1){		
        VECTOR3d diff = p0 - p1;
        SCALER distance = diff.norm();

        if (zs::abs((distance - expectedDistance)) > static_cast<SCALER>(1e-5) && (invMass0 + invMass1) > static_cast<SCALER>(1e-5)){
            VECTOR3d gradient = diff / (distance + static_cast<SCALER>(1e-6));
            SCALER denom = invMass0 + invMass1;
            SCALER lambda = (distance - expectedDistance) /denom;
            auto common = stiffness * lambda * gradient;
            // auto common = lambda * gradient;
            corr0 = -invMass0 * common;
            corr1 = invMass1 * common;
            // return false;
        }else{
            corr0 = VECTOR3d::uniform(0);
            corr1 = VECTOR3d::uniform(0);
        }

        return true;
    }


    // ----------------------------------------------------------------------------------------------
    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_VolumeConstraint(
        const VECTOR3d& p0, SCALER invMass0,
        const VECTOR3d& p1, SCALER invMass1,
        const VECTOR3d& p2, SCALER invMass2,
        const VECTOR3d& p3, SCALER invMass3,
        const SCALER restVolume,
        const SCALER stiffness,
        const SCALER dt,
        SCALER& lambda,
        VECTOR3d& corr0, VECTOR3d& corr1, VECTOR3d& corr2, VECTOR3d& corr3)
    {
        constexpr SCALER eps = (SCALER)1e-6;
        SCALER volume = static_cast<SCALER>(1.0 / 6.0) * (p1 - p0).cross(p2 - p0).dot(p3 - p0);

        corr0 = VECTOR3d::uniform(0); corr1 = VECTOR3d::uniform(0); corr2 = VECTOR3d::uniform(0); corr3 = VECTOR3d::uniform(0);

        VECTOR3d grad0 = (p1 - p2).cross(p3 - p2);
        VECTOR3d grad1 = (p2 - p0).cross(p3 - p0);
        VECTOR3d grad2 = (p0 - p1).cross(p3 - p1);
        VECTOR3d grad3 = (p1 - p0).cross(p2 - p0);

        SCALER K =
            invMass0 * grad0.l2NormSqr() +
            invMass1 * grad1.l2NormSqr() +
            invMass2 * grad2.l2NormSqr() +
            invMass3 * grad3.l2NormSqr();

        SCALER alpha = 0.0;
        if (stiffness != 0.0)
        {
            alpha = static_cast<SCALER>(1.0) / (stiffness * dt * dt);
            K += alpha;
        }

        if (zs::abs(K) < eps)
            return false;

        const SCALER C = volume - restVolume;
        const SCALER delta_lambda = -(C + alpha * lambda) / K;
        lambda += delta_lambda;

        corr0 = delta_lambda * invMass0 * grad0;
        corr1 = delta_lambda * invMass1 * grad1;
        corr2 = delta_lambda * invMass2 * grad2;
        corr3 = delta_lambda * invMass3 * grad3;

        return true;
    }

    template<typename VECTOR3d,typename SCALER = typename VECTOR3d::value_type>
    constexpr bool init_DihedralBendingConstraint(
        const VECTOR3d& p0,
        const VECTOR3d& p1,
        const VECTOR3d& p2,
        const VECTOR3d& p3,
        const SCALER& restAngleScale,
        SCALER& restAngle,
        SCALER& angleSign){
            VECTOR3d e = p3 - p2;
            SCALER elen = e.norm();
            if (elen < 1e-6)
                return false;

            angleSign = 1.0;

            restAngle = zs::dihedral_angle(p0,p2,p3,p1) * restAngleScale;
            return true;            
    }


    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_DihedralConstraint(
        const VECTOR3d &p0,const SCALER& invMass0,		
        const VECTOR3d &p1,const SCALER& invMass1,
        const VECTOR3d &p2,const SCALER& invMass2,
        const VECTOR3d &p3,const SCALER& invMass3,
        const SCALER& restAngle,
        // const SCALER& restAngleSign,
        const SCALER& stiffness,		
        VECTOR3d &corr0, VECTOR3d &corr1, VECTOR3d &corr2, VECTOR3d &corr3)
    {
        constexpr SCALER eps = static_cast<SCALER>(1e-6);
        if (invMass0 == 0.0 && invMass1 == 0.0)
            return false;

        auto angle = zs::dihedral_angle(p0,p2,p3,p1);
        auto grad = zs::dihedral_angle_gradient(p0,p2,p3,p1);
        VECTOR3d ds[4] = {};
        ds[0] = VECTOR3d{grad[0 * 3 + 0],grad[0 * 3 + 1],grad[0 * 3 + 2]};
        ds[2] = VECTOR3d{grad[1 * 3 + 0],grad[1 * 3 + 1],grad[1 * 3 + 2]};
        ds[3] = VECTOR3d{grad[2 * 3 + 0],grad[2 * 3 + 1],grad[2 * 3 + 2]};
        ds[1] = VECTOR3d{grad[3 * 3 + 0],grad[3 * 3 + 1],grad[3 * 3 + 2]};
        // for(int i = 0;i != 4;++i)
        //     ds[i] = VECTOR3d{grad[i * 3 + 0],grad[i * 3 + 1],grad[i * 3 + 2]};

        // for(int i = 0;i != 4;++i){
        //     printf("ds[%d] : %f %f %f\n",i,
        //         (float)ds[i][0],
        //         (float)ds[i][1],
        //         (float)ds[i][2]);
        // }


        // SCALER alpha = 0.0;
        // if (stiffness != 0.0)
        //     alpha = static_cast<SCALER>(1.0) / (stiffness * dt * dt);

        SCALER sum_normGradC = 
            invMass0 * ds[0].l2NormSqr() +
            invMass1 * ds[1].l2NormSqr() +
            invMass2 * ds[2].l2NormSqr() +
            invMass3 * ds[3].l2NormSqr();

        if(sum_normGradC < eps)
            return false;

        // compute impulse-based scaling factor
        // SCALER delta_lambda = (angle * angleSign - restAngle * restAngleSign + alpha * lambda) / sum_normGradC;
        auto C = (angle - restAngle);
        // auto al = alpha * lambda;
        // alpha = 0;
        // SCALER delta_lambda = -(C + alpha * lambda) / sum_normGradC;
        SCALER beta = -C * stiffness / sum_normGradC;
        // delta_lambda *= restAngleSign * angleSign;

        // auto ori_lambda = lambda;
        // lambda += delta_lambda;

        // printf("input angle : %f and restAngle : %f delta_lambda : %f : alpha * lambda : %f ori_lambda = %f\n",
        //     (float)angle,(float)restAngle,(float)delta_lambda,(float)al,(float)ori_lambda);

        corr0 = (beta * invMass0) * ds[0];
        corr1 = (beta * invMass1) * ds[1];
        corr2 = (beta * invMass2) * ds[2];
        corr3 = (beta * invMass3) * ds[3];
        return true;
    }


   /**
   *             v1 --- v3
   *            /  \    /
   *           /    \  /
   *          v2 --- v0
   */
    template<typename VECTOR3d,typename SCALER = typename VECTOR3d::value_type>
    constexpr bool solve_DihedralConstraint(
        const VECTOR3d& p0, const SCALER& invMass0,
        const VECTOR3d& p1, const SCALER& invMass1,
        const VECTOR3d& p2, const SCALER& invMass2,
        const VECTOR3d& p3, const SCALER& invMass3,
        const SCALER& restAngle,
        const SCALER& restAngleSign,
        const SCALER& xpbd_affliation,
        const SCALER& dt,
        SCALER& lambda,
        VECTOR3d& corr0, VECTOR3d& corr1, VECTOR3d& corr2, VECTOR3d& corr3)
    {  
        constexpr SCALER eps = static_cast<SCALER>(1e-6);
        if (invMass0 == 0.0 && invMass1 == 0.0)
            return false;

        auto angle = zs::dihedral_angle(p0,p2,p3,p1);
        auto grad = zs::dihedral_angle_gradient(p0,p2,p3,p1);
        VECTOR3d ds[4] = {};
        ds[0] = VECTOR3d{grad[0 * 3 + 0],grad[0 * 3 + 1],grad[0 * 3 + 2]};
        ds[2] = VECTOR3d{grad[1 * 3 + 0],grad[1 * 3 + 1],grad[1 * 3 + 2]};
        ds[3] = VECTOR3d{grad[2 * 3 + 0],grad[2 * 3 + 1],grad[2 * 3 + 2]};
        ds[1] = VECTOR3d{grad[3 * 3 + 0],grad[3 * 3 + 1],grad[3 * 3 + 2]};

        SCALER alpha = 0.0;
        if (xpbd_affliation != 0.0)
            alpha = static_cast<SCALER>(1.0) / (xpbd_affliation * dt * dt);

        SCALER sum_normGradC = 
            invMass0 * ds[0].l2NormSqr() +
            invMass1 * ds[1].l2NormSqr() +
            invMass2 * ds[2].l2NormSqr() +
            invMass3 * ds[3].l2NormSqr() + alpha;

        if(sum_normGradC < eps)
            return false;

        // compute impulse-based scaling factor
        // SCALER delta_lambda = (angle * angleSign - restAngle * restAngleSign + alpha * lambda) / sum_normGradC;
        auto C = (angle - restAngle);
        auto al = alpha * lambda;
        // alpha = 0;
        SCALER delta_lambda = -(C + alpha * lambda) / sum_normGradC;
        // delta_lambda *= restAngleSign * angleSign;

        // auto ori_lambda = lambda;
        lambda += delta_lambda;

        // printf("input angle : %f and restAngle : %f delta_lambda : %f : alpha * lambda : %f ori_lambda = %f\n",
        //     (float)angle,(float)restAngle,(float)delta_lambda,(float)al,(float)ori_lambda);

        corr0 = (delta_lambda * invMass0) * ds[0];
        corr1 = (delta_lambda * invMass1) * ds[1];
        corr2 = (delta_lambda * invMass2) * ds[2];
        corr3 = (delta_lambda * invMass3) * ds[3];
        return true;
    }


   /**
   *             v1 --- v3
   *            /  \    /
   *           /    \  /
   *          v2 --- v0
   */
   template<typename VECTOR3d,typename SCALER = typename VECTOR3d::value_type>
   constexpr bool solve_DihedralConstraint(
       const VECTOR3d& p0, const SCALER& invMass0,
       const VECTOR3d& p1, const SCALER& invMass1,
       const VECTOR3d& p2, const SCALER& invMass2,
       const VECTOR3d& p3, const SCALER& invMass3,
       const VECTOR3d& pp0,
       const VECTOR3d& pp1,
       const VECTOR3d& pp2,
       const VECTOR3d& pp3,
       const SCALER& restAngle,
       const SCALER& restAngleSign,
       const SCALER& xpbd_affliation,
       const SCALER& dt,
       const SCALER& kdamp_ratio,
       SCALER& lambda,
       VECTOR3d& corr0, VECTOR3d& corr1, VECTOR3d& corr2, VECTOR3d& corr3)
   {  
       constexpr SCALER eps = static_cast<SCALER>(1e-6);
       if (invMass0 == 0.0 && invMass1 == 0.0)
           return false;

       auto angle = zs::dihedral_angle(p0,p2,p3,p1);
       auto grad = zs::dihedral_angle_gradient(p0,p2,p3,p1);
       VECTOR3d ds[4] = {};
       ds[0] = VECTOR3d{grad[0 * 3 + 0],grad[0 * 3 + 1],grad[0 * 3 + 2]};
       ds[2] = VECTOR3d{grad[1 * 3 + 0],grad[1 * 3 + 1],grad[1 * 3 + 2]};
       ds[3] = VECTOR3d{grad[2 * 3 + 0],grad[2 * 3 + 1],grad[2 * 3 + 2]};
       ds[1] = VECTOR3d{grad[3 * 3 + 0],grad[3 * 3 + 1],grad[3 * 3 + 2]};

       SCALER alpha = 0.0;
       if (xpbd_affliation != 0.0)
           alpha = static_cast<SCALER>(1.0) / (xpbd_affliation * dt * dt);

        SCALER dsum = 0.0;
        SCALER gamma = 1.0;
        if(kdamp_ratio > 0) {
            auto beta = kdamp_ratio * dt * dt;
            gamma = alpha * beta / dt;

            dsum += gamma * ds[0].dot(p0 - pp0);
            dsum += gamma * ds[1].dot(p1 - pp1);
            dsum += gamma * ds[2].dot(p2 - pp2);
            dsum += gamma * ds[3].dot(p3 - pp3);
        }

       SCALER wsum = 
           invMass0 * ds[0].l2NormSqr() +
           invMass1 * ds[1].l2NormSqr() +
           invMass2 * ds[2].l2NormSqr() +
           invMass3 * ds[3].l2NormSqr();
        
        wsum = (gamma + static_cast<SCALER>(1.0)) * wsum + alpha;

       if(wsum < eps)
           return false;

       // compute impulse-based scaling factor
       // SCALER delta_lambda = (angle * angleSign - restAngle * restAngleSign + alpha * lambda) / sum_normGradC;
       auto C = (angle - restAngle);
       auto al = alpha * lambda;
       // alpha = 0;
       SCALER delta_lambda = -(C + alpha * lambda + dsum) / wsum;
       // delta_lambda *= restAngleSign * angleSign;

       // auto ori_lambda = lambda;
       lambda += delta_lambda;

       // printf("input angle : %f and restAngle : %f delta_lambda : %f : alpha * lambda : %f ori_lambda = %f\n",
       //     (float)angle,(float)restAngle,(float)delta_lambda,(float)al,(float)ori_lambda);

       corr0 = (delta_lambda * invMass0) * ds[0];
       corr1 = (delta_lambda * invMass1) * ds[1];
       corr2 = (delta_lambda * invMass2) * ds[2];
       corr3 = (delta_lambda * invMass3) * ds[3];
       return true;
   }

// ----------------------------------------------------------------------------------------------
    template<typename VECTOR3d,typename MATRIX4d,typename SCALER = typename VECTOR3d::value_type>
    constexpr bool init_IsometricBendingConstraint(
        const VECTOR3d& p0,
        const VECTOR3d& p1,
        const VECTOR3d& p2,
        const VECTOR3d& p3,
        MATRIX4d& Q,
        SCALER& C0)
    {
        // Compute matrix Q for quadratic bending
        const VECTOR3d x[4] = { p2, p3, p0, p1 };
        // Q = MATRIX4d::uniform(0);

        // const auto e0 = x[1].template cast<double>() - x[0].template cast<double>();
        // const auto e1 = x[2].template cast<double>() - x[0].template cast<double>();
        // const auto e2 = x[3].template cast<double>() - x[0].template cast<double>();
        // const auto e3 = x[2].template cast<double>() - x[1].template cast<double>();
        // const auto e4 = x[3].template cast<double>() - x[1].template cast<double>();

        const auto e0 = x[1].template cast<double>() - x[0].template cast<double>();
        const auto e1 = x[2].template cast<double>() - x[0].template cast<double>();
        const auto e2 = x[3].template cast<double>() - x[0].template cast<double>();
        const auto e3 = x[2].template cast<double>() - x[1].template cast<double>();
        const auto e4 = x[3].template cast<double>() - x[1].template cast<double>();


        // auto e0 = e0_.template cast<double>();
        // auto e1 = e1_.template cast<double>();
        // auto e2 = e2_.template cast<double>();
        // auto e3 = e3_.template cast<double>();
        // auto e4 = e4_.template cast<double>();

        // printf("init isometric bending energy : %f %f %f %f\n",
        //     (float)p0.norm(),
        //     (float)p1.norm(),
        //     (float)p2.norm(),
        //     (float)p3.norm());

        const double c01 = LSL_GEO::cotTheta(e0, e1);
        const double c02 = LSL_GEO::cotTheta(e0, e2);
        const double c03 = LSL_GEO::cotTheta(-e0, e3);
        const double c04 = LSL_GEO::cotTheta(-e0, e4);

        const double A0 = static_cast<double>(0.5) * (e0.cross(e1)).norm();
        const double A1 = static_cast<double>(0.5) * (e0.cross(e2)).norm();

        const double coef = static_cast<double>(-3.0 / 2.0) /  (A0 + A1);
        const double K[4] = { c03 + c04, c01 + c02, -c01 - c03, -c02 - c04 };
        const double K2[4] = { coef * K[0], coef * K[1], coef * K[2], coef * K[3] };

        for (unsigned char j = 0; j < 4; j++)
        {
            for (unsigned char k = 0; k < j; k++)
            {
                Q(j, k) = Q(k, j) = (SCALER)(K[j] * K2[k]);
            }
            Q(j, j) = (SCALER)(K[j] * K2[j]);
        }

        C0 = 0.0;
        for (unsigned char k = 0; k < 4; k++)
            for (unsigned char j = 0; j < 4; j++)
                C0 += Q(j, k) * (x[k].dot(x[j]));
        C0 *= static_cast<SCALER>(0.5);        

        return true;
    }
// ----------------------------------------------------------------------------------------------
    template<typename VECTOR3d,typename SCALER,typename MATRIX4d>
    constexpr bool solve_IsometricBendingConstraint(
        const VECTOR3d& p0, const SCALER& invMass0,
        const VECTOR3d& p1, const SCALER& invMass1,
        const VECTOR3d& p2, const SCALER& invMass2,
        const VECTOR3d& p3, const SCALER& invMass3,
        const MATRIX4d& Q,
        const SCALER& stiffness,
        const SCALER& dt,
        const SCALER& C0,
        SCALER& lambda,
        VECTOR3d& corr0, VECTOR3d& corr1, VECTOR3d& corr2, VECTOR3d& corr3)
    {
        constexpr SCALER eps = static_cast<SCALER>(1e-4);
        const VECTOR3d x[4] = { p2, p3, p0, p1 };
        SCALER invMass[4] = { invMass2, invMass3, invMass0, invMass1 };

        SCALER C = 0.0;
        for (unsigned char k = 0; k < 4; k++)
            for (unsigned char j = 0; j < 4; j++)
                C += Q(j, k) * (x[k].dot(x[j]));
        C *= static_cast<SCALER>(0.5);
        C -= C0;

        // printf("isometric_bending_energy : %f\n",(float)energy);

        // printf("solve isometric bending energy : %f %f %f %f\n",
        //     (float)p0.norm(),
        //     (float)p1.norm(),
        //     (float)p2.norm(),
        //     (float)p3.norm());

        VECTOR3d gradC[4] = {};
        gradC[0] = VECTOR3d::uniform(0);
        gradC[1] = VECTOR3d::uniform(0);
        gradC[2] = VECTOR3d::uniform(0);
        gradC[3] = VECTOR3d::uniform(0);
        for (unsigned char k = 0; k < 4; k++)
            for (unsigned char j = 0; j < 4; j++)
                gradC[j] += Q(j, k) * x[k];


        SCALER sum_normGradC = 0.0;
        for (unsigned int j = 0; j < 4; j++)
        {
            // compute sum of squared gradient norms
            if (invMass[j] != 0.0)
                sum_normGradC += invMass[j] * gradC[j].l2NormSqr();
        }

        SCALER alpha = 0.0;
        if (stiffness != 0.0)
        {
            alpha = static_cast<SCALER>(1.0) / (stiffness * dt * dt);
            sum_normGradC += alpha;
        }

        // exit early if required


        if (sum_normGradC > eps)
        {
            // compute impulse-based scaling factor
            SCALER delta_lambda = -(C + alpha * lambda) / sum_normGradC;
            lambda += delta_lambda;

            corr0 = (delta_lambda * invMass[2]) * gradC[2];
            corr1 = (delta_lambda * invMass[3]) * gradC[3];
            corr2 = (delta_lambda * invMass[0]) * gradC[0];
            corr3 = (delta_lambda * invMass[1]) * gradC[1];
            // printf("update corr for iso_bending energy\n");

            // auto gradCSum = (float)0;
            // for(int i = 0;i != 4;++i)
            //     gradCSum += gradC[i].norm();
            // if(gradCSum > 1e-4) {
            //     // printf("gradC : %f %f %f %f corr : %f %f %f %f\n",
            //     //     (float)gradC[0].norm(),(float)gradC[1].norm(),(float)gradC[2].norm(),(float)gradC[3].norm(),
            //     //     (float)corr0.norm(),(float)corr1.norm(),(float)corr2.norm(),(float)corr3.norm());
            // }

            return true;
        }else {
            return false;
        }
    }


#if 0
    template<typename VECTOR3d,typename SCALER,typename MATRIX4d>
    constexpr bool solve_IsometricBendingConstraint(
        const VECTOR3d& p0, SCALER invMass0,
        const VECTOR3d& p1, SCALER invMass1,
        const VECTOR3d& p2, SCALER invMass2,
        const VECTOR3d& p3, SCALER invMass3,
        const MATRIX4d& Q,
        const SCALER& stiffness,
        const SCALER& C0,
        VECTOR3d& corr0, VECTOR3d& corr1, VECTOR3d& corr2, VECTOR3d& corr3){
        constexpr SCALER eps = static_cast<SCALER>(1e-6);
        const VECTOR3d x[4] = { p2, p3, p0, p1 };
        SCALER invMass[4] = { invMass2, invMass3, invMass0, invMass1 };

        SCALER C = 0.0;
        for (unsigned char k = 0; k < 4; k++)
            for (unsigned char j = 0; j < 4; j++)
                C += Q(j, k) * (x[k].dot(x[j]));
        C *= 0.5;
        C -= C0;

        VECTOR3d gradC[4] = {};
        gradC[0] = VECTOR3d::uniform(0);
        gradC[1] = VECTOR3d::uniform(0);
        gradC[2] = VECTOR3d::uniform(0);
        gradC[3] = VECTOR3d::uniform(0);
        for (unsigned char k = 0; k < 4; k++)
            for (unsigned char j = 0; j < 4; j++)
                gradC[j] += Q(j, k) * x[k];


        SCALER sum_normGradC = 0.0;
        for (unsigned int j = 0; j < 4; j++)
        {
            // compute sum of squared gradient norms
            if (invMass[j] != 0.0)
                sum_normGradC += invMass[j] * gradC[j].l2NormSqr();
        }

        // exit early if required
        if (sum_normGradC > eps)
        {
            // compute impulse-based scaling factor
            const SCALER s = -(energy)/ sum_normGradC;

            corr0 = stiffness  * (s * invMass[2]) * gradC[2];
            corr1 = stiffness  * (s * invMass[3]) * gradC[3];
            corr2 = stiffness  * (s * invMass[0]) * gradC[0];
            corr3 = stiffness  * (s * invMass[1]) * gradC[1];


            return true;
        }else {
            corr0 = VECTOR3d::uniform(0);
            corr1 = VECTOR3d::uniform(0);
            corr2 = VECTOR3d::uniform(0);
            corr3 = VECTOR3d::uniform(0);
        }
        return false;
    }
#endif

    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_PlaneConstraint(
        const VECTOR3d& p, SCALER invMass,
        const VECTOR3d& root,
        const VECTOR3d& nrm,
        const SCALER& thickness,
        const SCALER& stiffness,
        const SCALER& dt,
        SCALER& lambda,
        VECTOR3d& dp) {
            SCALER C = (p - root).dot(nrm) - thickness;
            if(C > static_cast<SCALER>(1e-6)) {
                dp = VECTOR3d::uniform(0);
                return true;
            }

            SCALER K = invMass * nrm.l2NormSqr();

            SCALER alpha = 0.0;
            if(stiffness != 0.0) {
                alpha = static_cast<SCALER>(1.0) / (stiffness * dt * dt);
                K += alpha;                
            }

            SCALER Kinv = 0.0;
            if(zs::abs(K) > static_cast<SCALER>(1e-6))
                Kinv = static_cast<SCALER>(1.0) / K;
            else                       
            {
                dp = VECTOR3d::uniform(0);
                return true;
            }     

            const SCALER delta_lambda = -Kinv * (C + alpha * lambda);           
            lambda += delta_lambda;
            const VECTOR3d pt = nrm * delta_lambda;

            dp = invMass * pt;
            return true;
    }

// ----------------------------------------------------------------------------------------------
    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_TrianglePointDistanceConstraint(
        const VECTOR3d &p, SCALER invMass,
        const VECTOR3d &p0, SCALER invMass0,
        const VECTOR3d &p1, SCALER invMass1,
        const VECTOR3d &p2, SCALER invMass2,
        const SCALER restDist,
        const SCALER compressionStiffness,
        const SCALER stretchStiffness,
        SCALER& lambda,
        VECTOR3d &corr, VECTOR3d &corr0, VECTOR3d &corr1, VECTOR3d &corr2)
    {
        // find barycentric coordinates of closest point on triangle

        SCALER b0 = static_cast<SCALER>(1.0 / 3.0);		// for singular case
        SCALER b1 = b0;
        SCALER b2 = b0;

        VECTOR3d d1 = p1 - p0;
        VECTOR3d d2 = p2 - p0;
        VECTOR3d pp0 = p - p0;
        SCALER a = d1.dot(d1);
        SCALER b = d2.dot(d1);
        SCALER c = pp0.dot(d1);
        SCALER d = b;
        SCALER e = d2.dot(d2);
        SCALER f = pp0.dot(d2);
        SCALER det = a*e - b*d;

        if (det != 0.0) {
            SCALER s = (c*e - b*f) / det;
            SCALER t = (a*f - c*d) / det;
            b0 = static_cast<SCALER>(1.0) - s - t;		// inside triangle
            b1 = s;
            b2 = t;
            if (b0 < 0.0) {		// on edge 1-2
                VECTOR3d d = p2 - p1;
                SCALER d2 = d.dot(d);
                SCALER t = (d2 == static_cast<SCALER>(0.0)) ? static_cast<SCALER>(0.5) : d.dot(p - p1) / d2;
                if (t < 0.0) t = 0.0;	// on point 1
                if (t > 1.0) t = 1.0;	// on point 2
                b0 = 0.0;
                b1 = (static_cast<SCALER>(1.0) - t);
                b2 = t;
            }
            else if (b1 < 0.0) {	// on edge 2-0
                VECTOR3d d = p0 - p2;
                SCALER d2 = d.dot(d);
                SCALER t = (d2 == static_cast<SCALER>(0.0)) ? static_cast<SCALER>(0.5) : d.dot(p - p2) / d2;
                if (t < 0.0) t = 0.0;	// on point 2
                if (t > 1.0) t = 1.0; // on point 0
                b1 = 0.0;
                b2 = (static_cast<SCALER>(1.0) - t);
                b0 = t;
            }
            else if (b2 < 0.0) {	// on edge 0-1
                VECTOR3d d = p1 - p0;
                SCALER d2 = d.dot(d);
                SCALER t = (d2 == static_cast<SCALER>(0.0)) ? static_cast<SCALER>(0.5) : d.dot(p - p0) / d2;
                if (t < 0.0) t = 0.0;	// on point 0
                if (t > 1.0) t = 1.0;	// on point 1
                b2 = 0.0;
                b0 = (static_cast<SCALER>(1.0) - t);
                b1 = t;
            }
        }
        VECTOR3d q = p0 * b0 + p1 * b1 + p2 * b2;
        VECTOR3d n = p - q;
        SCALER dist = n.norm();
        n.normalize();
        SCALER C = dist - restDist;
        VECTOR3d grad = n;
        VECTOR3d grad0 = -n * b0;
        VECTOR3d grad1 = -n * b1;
        VECTOR3d grad2 = -n * b2;

        SCALER s = invMass + invMass0 * b0*b0 + invMass1 * b1*b1 + invMass2 * b2*b2;
        if (s == 0.0)
            return false;

        s = C / s;
        if (C < 0.0)
            s *= compressionStiffness;
        else
            s *= stretchStiffness;

        if (s == 0.0)
            return false;

        corr = -s * invMass * grad;
        corr0 = -s * invMass0 * grad0;
        corr1 = -s * invMass1 * grad1;
        corr2 = -s * invMass2 * grad2;
        return true;
    }

    // ----------------------------------------------------------------------------------------------
    template<typename VECTOR3d,typename SCALER>
    constexpr bool solve_EdgeEdgeDistanceConstraint(
        const VECTOR3d &p0, SCALER invMass0,
        const VECTOR3d &p1, SCALER invMass1,
        const VECTOR3d &p2, SCALER invMass2,
        const VECTOR3d &p3, SCALER invMass3,
        const SCALER restDist,
        const SCALER compressionStiffness,
        const SCALER stretchStiffness,
        SCALER& lambda,
        VECTOR3d &corr0, VECTOR3d &corr1, VECTOR3d &corr2, VECTOR3d &corr3)
    {
        VECTOR3d d0 = p1 - p0;
        VECTOR3d d1 = p3 - p2;

        SCALER a = d0.l2NormSqr();
        SCALER b = -d0.dot(d1);
        SCALER c = d0.dot(d1);
        SCALER d = -d1.l2NormSqr();
        SCALER e = (p2 - p0).dot(d0);
        SCALER f = (p2 - p0).dot(d1);
        SCALER det = a*d - b*c;
        SCALER s, t;
        if (det != 0.0) {
            det = static_cast<SCALER>(1.0) / det;
            s = (e*d - b*f) * det;
            t = (a*f - e*c) * det;
        }
        else {	// d0 and d1 parallel
            SCALER s0 = p0.dot(d0);
            SCALER s1 = p1.dot(d0);
            SCALER t0 = p2.dot(d0);
            SCALER t1 = p3.dot(d0);
            bool flip0 = false;
            bool flip1 = false;

            if (s0 > s1) { SCALER f = s0; s0 = s1; s1 = f; flip0 = true; }
            if (t0 > t1) { SCALER f = t0; t0 = t1; t1 = f; flip1 = true; }

            if (s0 >= t1) {
                s = !flip0 ? static_cast<SCALER>(0.0) : static_cast<SCALER>(1.0);
                t = !flip1 ? static_cast<SCALER>(1.0) : static_cast<SCALER>(0.0);
            }
            else if (t0 >= s1) {
                s = !flip0 ? static_cast<SCALER>(1.0) : static_cast<SCALER>(0.0);
                t = !flip1 ? static_cast<SCALER>(0.0) : static_cast<SCALER>(1.0);
            }
            else {		// overlap
                SCALER mid = (s0 > t0) ? (s0 + t1) * static_cast<SCALER>(0.5) : (t0 + s1) * static_cast<SCALER>(0.5);
                s = (s0 == s1) ? static_cast<SCALER>(0.5) : (mid - s0) / (s1 - s0);
                t = (t0 == t1) ? static_cast<SCALER>(0.5) : (mid - t0) / (t1 - t0);
            }
        }
        if (s < 0.0) s = 0.0;
        if (s > 1.0) s = 1.0;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;

        SCALER b0 = static_cast<SCALER>(1.0) - s;
        SCALER b1 = s;
        SCALER b2 = static_cast<SCALER>(1.0) - t;
        SCALER b3 = t;

        VECTOR3d q0 = p0 * b0 + p1 * b1;
        VECTOR3d q1 = p2 * b2 + p3 * b3;
        VECTOR3d n = q0 - q1;
        SCALER dist = n.norm();
        n.normalize();
        SCALER C = dist - restDist;
        VECTOR3d grad0 = n * b0;
        VECTOR3d grad1 = n * b1;
        VECTOR3d grad2 = -n * b2;
        VECTOR3d grad3 = -n * b3;

        s = invMass0 * b0*b0 + invMass1 * b1*b1 + invMass2 * b2*b2 + invMass3 * b3*b3;
        if (s == 0.0)
            return false;

        s = C / s;
        if (C < 0.0)
            s *= compressionStiffness;
        else
            s *= stretchStiffness;

        if (s == 0.0)
            return false;

        corr0 = -s * invMass0 * grad0;
        corr1 = -s * invMass1 * grad1;
        corr2 = -s * invMass2 * grad2;
        corr3 = -s * invMass3 * grad3;
        return true;
    }

// FOR ELASTIC RODS SIMULATION
// ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename SCALER,typename QUATERNION>
//     constexpr bool solve_StretchShearConstraint(
//         const VECTOR3d& p0, SCALER invMass0,
//         const VECTOR3d& p1, SCALER invMass1,
//         const QUATERNION& q0, SCALER invMassq0,
//         const VECTOR3d& stretchingAndShearingKs,
//         const SCALER restLength,
//         VECTOR3d& corr0, VECTOR3d& corr1, QUATERNION& corrq0)
//     {
//         VECTOR3d d3;	//third director d3 = q0 * e_3 * q0_conjugate
//         d3[0] = static_cast<SCALER>(2.0) * (q0.x() * q0.z() + q0.w() * q0.y());
//         d3[1] = static_cast<SCALER>(2.0) * (q0.y() * q0.z() - q0.w() * q0.x());
//         d3[2] = q0.w() * q0.w() - q0.x() * q0.x() - q0.y() * q0.y() + q0.z() * q0.z();

//         VECTOR3d gamma = (p1 - p0) / restLength - d3;
//         gamma /= (invMass1 + invMass0) / restLength + invMassq0 * static_cast<SCALER>(4.0)*restLength + eps;

//         if (std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[1]) < eps && std::abs(stretchingAndShearingKs[0] - stretchingAndShearingKs[2]) < eps)	//all Ks are approx. equal
//             for (int i = 0; i<3; i++) gamma[i] *= stretchingAndShearingKs[i];
//         else	//diffenent stretching and shearing Ks. Transform diag(Ks[0], Ks[1], Ks[2]) into world space using Ks_w = R(q0) * diag(Ks[0], Ks[1], Ks[2]) * R^T(q0) and multiply it with gamma
//         {
//             MATRIX3d R = q0.toRotationMatrix();
//             gamma = (R.transpose() * gamma).eval();
//             for (int i = 0; i<3; i++) gamma[i] *= stretchingAndShearingKs[i];
//             gamma = (R * gamma).eval();
//         }

//         corr0 = invMass0 * gamma;
//         corr1 = -invMass1 * gamma;

//         QUATERNION q_e_3_bar(q0.z(), -q0.y(), q0.x(), -q0.w());	//compute q*e_3.conjugate (cheaper than quaternion product)
//         corrq0 = QUATERNION(0.0, gamma.x(), gamma.y(), gamma.z()) * q_e_3_bar;
//         corrq0.coeffs() *= static_cast<SCALER>(2.0) * invMassq0 * restLength;

//         return true;
//     }

// // ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename SCALER,typename QUATERNION>
//     constexpr bool solve_BendTwistConstraint(
//         const QUATERNION& q0, SCALER invMassq0,
//         const QUATERNION& q1, SCALER invMassq1,
//         const VECTOR3d& bendingAndTwistingKs,
//         const QUATERNION& restDarbouxVector,
//         QUATERNION& corrq0, QUATERNION& corrq1)
//     {
//         QUATERNION omega = q0.conjugate() * q1;   //darboux vector

//         QUATERNION omega_plus;
//         omega_plus.coeffs() = omega.coeffs() + restDarbouxVector.coeffs();     //delta Omega with -Omega_0
//         omega.coeffs() = omega.coeffs() - restDarbouxVector.coeffs();                 //delta Omega with + omega_0
//         if (omega.l2NormSqr() > omega_plus.l2NormSqr()) omega = omega_plus;

//         for (int i = 0; i < 3; i++) omega.coeffs()[i] *= bendingAndTwistingKs[i] / (invMassq0 + invMassq1 + static_cast<SCALER>(1.0e-6));
//         omega.w() = 0.0;    //discrete Darboux vector does not have vanishing scalar part

//         corrq0 = q1 * omega;
//         corrq1 = q0 * omega;
//         corrq0.coeffs() *= invMassq0;
//         corrq1.coeffs() *= -invMassq1;
//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename SCALER>
//     constexpr bool solve_PerpendiculaBisectorConstraint(
//         const VECTOR3d &p0, SCALER invMass0,
//         const VECTOR3d &p1, SCALER invMass1,
//         const VECTOR3d &p2, SCALER invMass2,
//         const SCALER stiffness,
//         VECTOR3d &corr0, VECTOR3d &corr1, VECTOR3d &corr2)
//     {
//         const VECTOR3d pm = 0.5 * (p0 + p1);
//         const VECTOR3d p0p2 = p0 - p2;
//         const VECTOR3d p2p1 = p2 - p1;
//         const VECTOR3d p1p0 = p1 - p0;
//         const VECTOR3d p2pm = p2 - pm;

//         SCALER wSum = invMass0 * p0p2.l2NormSqr() + invMass1 * p2p1.l2NormSqr() + invMass2 * p1p0.l2NormSqr();
//         if (wSum < eps)
//             return false;

//         const SCALER lambda = stiffness * p2pm.dot(p1p0) / wSum;

//         corr0 = -invMass0 * lambda * p0p2;
//         corr1 = -invMass1 * lambda * p2p1;
//         corr2 = -invMass2 * lambda * p1p0;

//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename SCALER>
//     constexpr bool solve_GhostPointEdgeDistanceConstraint(
//         const VECTOR3d& p0, SCALER invMass0,
//         const VECTOR3d& p1, SCALER invMass1,
//         const VECTOR3d& p2, SCALER invMass2,
//         const SCALER stiffness, 
//         const SCALER ghostEdgeRestLength,
//         VECTOR3d& corr0, VECTOR3d&  corr1, VECTOR3d&  corr2)
//     {
//         // Ghost-Edge constraint
//         VECTOR3d pm = 0.5 * (p0 + p1);
//         VECTOR3d p2pm = p2 - pm;
//         SCALER wSum = static_cast<SCALER>(0.25) * invMass0 + static_cast<SCALER>(0.25) * invMass1 + static_cast<SCALER>(1.0) * invMass2;

//         if (wSum < eps)
//             return false;

//         SCALER p2pm_mag = p2pm.norm();
//         p2pm *= static_cast<SCALER>(1.0) / p2pm_mag;

//         const SCALER lambda = stiffness * (p2pm_mag - ghostEdgeRestLength) / wSum;

//         corr0 = 0.5 * invMass0 * lambda * p2pm;
//         corr1 = 0.5 * invMass1 * lambda * p2pm;
//         corr2 = -1.0 * invMass2 * lambda * p2pm;

//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename SCALER>
//     constexpr bool solve_DarbouxVectorConstraint(
//         const VECTOR3d& p0, SCALER invMass0,
//         const VECTOR3d& p1, SCALER invMass1,
//         const VECTOR3d& p2, SCALER invMass2,
//         const VECTOR3d& p3, SCALER invMass3,
//         const VECTOR3d& p4, SCALER invMass4,
//         const VECTOR3d& bendingAndTwistingKs,
//         const SCALER midEdgeLength,
//         const VECTOR3d& restDarbouxVector,
//         VECTOR3d& corr0, VECTOR3d&  corr1, VECTOR3d&  corr2, VECTOR3d&  corr3, VECTOR3d& corr4)
//     {
//         //  Single rod element:
//         //      3   4		//ghost points
//         //		|	|
//         //  --0---1---2--	// rod points

//         VECTOR3d darboux_vector;
//         MATRIX3d d0, d1;

//         computeMaterialFrame(p0, p1, p3, d0);
//         computeMaterialFrame(p1, p2, p4, d1);

//         computeDarbouxVector(d0, d1, midEdgeLength, darboux_vector);

//         MATRIX3d dajpi[3][3];
//         computeMaterialFrameDerivative(p0, p1, p3, d0,
//             dajpi[0][0], dajpi[0][1], dajpi[0][2],
//             dajpi[1][0], dajpi[1][1], dajpi[1][2],
//             dajpi[2][0], dajpi[2][1], dajpi[2][2]);

//         MATRIX3d dbjpi[3][3];
//         computeMaterialFrameDerivative(p1, p2, p4, d1,
//             dbjpi[0][0], dbjpi[0][1], dbjpi[0][2],
//             dbjpi[1][0], dbjpi[1][1], dbjpi[1][2],
//             dbjpi[2][0], dbjpi[2][1], dbjpi[2][2]);

//         MATRIX3d constraint_jacobian[5];
//         computeDarbouxGradient(
//             darboux_vector, midEdgeLength, d0, d1, 
//             dajpi, dbjpi, 
//             //bendingAndTwistingKs,
//             constraint_jacobian[0],
//             constraint_jacobian[1],
//             constraint_jacobian[2],
//             constraint_jacobian[3],
//             constraint_jacobian[4]);

//         const VECTOR3d constraint_value(bendingAndTwistingKs[0] * (darboux_vector[0] - restDarbouxVector[0]),
//                                 bendingAndTwistingKs[1] * (darboux_vector[1] - restDarbouxVector[1]),
//                                 bendingAndTwistingKs[2] * (darboux_vector[2] - restDarbouxVector[2]));

//         MATRIX3d factor_matrix;
//         factor_matrix = VECTOR3d::uniform(0);

//         MATRIX3d tmp_mat;
//         SCALER invMasses[]{ invMass0, invMass1, invMass2, invMass3, invMass4 };
//         for (int i = 0; i < 5; ++i)
//         {
//             tmp_mat = constraint_jacobian[i].transpose() * constraint_jacobian[i];
//             tmp_mat.col(0) *= invMasses[i];
//             tmp_mat.col(1) *= invMasses[i];
//             tmp_mat.col(2) *= invMasses[i];

//             factor_matrix += tmp_mat;
//         }

//         VECTOR3d dp[5];
//         tmp_mat = factor_matrix.inverse();

//         for (int i = 0; i < 5; ++i)
//         {
//             constraint_jacobian[i].col(0) *= invMasses[i];
//             constraint_jacobian[i].col(1) *= invMasses[i];
//             constraint_jacobian[i].col(2) *= invMasses[i];
//             dp[i] = -(constraint_jacobian[i]) * (tmp_mat * constraint_value);
//         }

//         corr0 = dp[0];
//         corr1 = dp[1];
//         corr2 = dp[2];
//         corr3 = dp[3];
//         corr4 = dp[4];

//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename MATRIX3d>
//     constexpr bool computeMaterialFrame(
//         const VECTOR3d& p0, 
//         const VECTOR3d& p1, 
//         const VECTOR3d& p2, 
//         MATRIX3d& frame)
//     {
//         frame.col(2) = (p1 - p0);
//         frame.col(2).normalize();

//         frame.col(1) = (frame.col(2).cross(p2 - p0));
//         frame.col(1).normalize();

//         frame.col(0) = frame.col(1).cross(frame.col(2));
//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename SCALER,typename VECTOR3d,typename MATRIX3d>
//     constexpr bool computeDarbouxVector(const MATRIX3d& dA, const MATRIX3d& dB, const SCALER mid_edge_length, VECTOR3d& darboux_vector)
//     {
//         SCALER factor = static_cast<SCALER>(1.0) + dA.col(0).dot(dB.col(0)) + dA.col(1).dot(dB.col(1)) + dA.col(2).dot(dB.col(2));

//         factor = static_cast<SCALER>(2.0) / (mid_edge_length * factor);

//         for (int c = 0; c < 3; ++c)
//         {
//             const int i = permutation[c][0];
//             const int j = permutation[c][1];
//             const int k = permutation[c][2];
//             darboux_vector[i] = dA.col(j).dot(dB.col(k)) - dA.col(k).dot(dB.col(j));
//         }
//         darboux_vector *= factor;
//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename VECTOR3d,typename MATRIX3d>
//     constexpr bool computeMaterialFrameDerivative(
//         const VECTOR3d& p0, const VECTOR3d& p1, const VECTOR3d& p2, const MATRIX3d& d,
//         MATRIX3d& d1p0, MATRIX3d& d1p1, MATRIX3d& d1p2,
//         MATRIX3d& d2p0, MATRIX3d& d2p1, MATRIX3d& d2p2,
//         MATRIX3d& d3p0, MATRIX3d& d3p1, MATRIX3d& d3p2)
//     {
//         //////////////////////////////////////////////////////////////////////////
//         // d3pi
//         //////////////////////////////////////////////////////////////////////////
//         const VECTOR3d p01 = p1 - p0;
//         SCALER length_p01 = p01.norm();

//         d3p0.col(0) = d.col(2)[0] * d.col(2);
//         d3p0.col(1) = d.col(2)[1] * d.col(2);
//         d3p0.col(2) = d.col(2)[2] * d.col(2);

//         d3p0.col(0)[0] -= 1.0;
//         d3p0.col(1)[1] -= 1.0;
//         d3p0.col(2)[2] -= 1.0;

//         d3p0.col(0) *= (static_cast<SCALER>(1.0) / length_p01);
//         d3p0.col(1) *= (static_cast<SCALER>(1.0) / length_p01);
//         d3p0.col(2) *= (static_cast<SCALER>(1.0) / length_p01);

//         d3p1.col(0) = -d3p0.col(0);
//         d3p1.col(1) = -d3p0.col(1);
//         d3p1.col(2) = -d3p0.col(2);

//         d3p2.col(0) = VECTOR3d::uniform(0);
//         d3p2.col(1) = VECTOR3d::uniform(0);
//         d3p2.col(2) = VECTOR3d::uniform(0);

//         //////////////////////////////////////////////////////////////////////////
//         // d2pi
//         //////////////////////////////////////////////////////////////////////////
//         const VECTOR3d p02 = p2 - p0;
//         const VECTOR3d p01_cross_p02 = p01.cross(p02);

//         const SCALER length_cross = p01_cross_p02.norm();

//         MATRIX3d mat;
//         mat.col(0) = d.col(1)[0] * d.col(1);
//         mat.col(1) = d.col(1)[1] * d.col(1);
//         mat.col(2) = d.col(1)[2] * d.col(1);

//         mat.col(0)[0] -= 1.0;
//         mat.col(1)[1] -= 1.0;
//         mat.col(2)[2] -= 1.0;

//         mat.col(0) *= (-static_cast<SCALER>(1.0) / length_cross);
//         mat.col(1) *= (-static_cast<SCALER>(1.0) / length_cross);
//         mat.col(2) *= (-static_cast<SCALER>(1.0) / length_cross);

//         MATRIX3d product_matrix;
//         LSL_GEO::crossProductMatrix(p2 - p1, product_matrix);
//         d2p0 = mat * product_matrix;

//         LSL_GEO::crossProductMatrix(p0 - p2, product_matrix);
//         d2p1 = mat * product_matrix;

//         LSL_GEO::crossProductMatrix(p1 - p0, product_matrix);
//         d2p2 = mat * product_matrix;

//         //////////////////////////////////////////////////////////////////////////
//         // d1pi
//         //////////////////////////////////////////////////////////////////////////
//         MATRIX3d product_mat_d3;
//         MATRIX3d product_mat_d2;
//         LSL_GEO::crossProductMatrix(d.col(2), product_mat_d3);
//         LSL_GEO::crossProductMatrix(d.col(1), product_mat_d2);

//         d1p0 = product_mat_d2 * d3p0 - product_mat_d3 * d2p0;
//         d1p1 = product_mat_d2 * d3p1 - product_mat_d3 * d2p1;
//         d1p2 = -product_mat_d3 * d2p2;
//         return true;
//     }

//     // ----------------------------------------------------------------------------------------------
//     template<typename SCALER,typename VECTOR3d,typename MATRIX3d>
//     constexpr bool computeDarbouxGradient(
//         const VECTOR3d& darboux_vector, const SCALER length,
//         const MATRIX3d& da, const MATRIX3d& db,
//         const MATRIX3d dajpi[3][3], const MATRIX3d dbjpi[3][3],
//         //const VECTOR3d& bendAndTwistKs,
//         MATRIX3d& omega_pa, MATRIX3d& omega_pb, MATRIX3d& omega_pc, MATRIX3d& omega_pd, MATRIX3d& omega_pe
//         )
//     {
//         SCALER X = static_cast<SCALER>(1.0) + da.col(0).dot(db.col(0)) + da.col(1).dot(db.col(1)) + da.col(2).dot(db.col(2));
//         X = static_cast<SCALER>(2.0) / (length * X);

//         for (int c = 0; c < 3; ++c) 
//         {
//             const int i = permutation[c][0];
//             const int j = permutation[c][1];
//             const int k = permutation[c][2];
//             // pa
//             {
//                 VECTOR3d term1(0,0,0);
//                 VECTOR3d term2(0,0,0);
//                 VECTOR3d tmp(0,0,0);

//                 // first term
//                 term1 = dajpi[j][0].transpose() * db.col(k);
//                 tmp =   dajpi[k][0].transpose() * db.col(j);
//                 term1 = term1 - tmp;
//                 // second term
//                 for (int n = 0; n < 3; ++n) 
//                 {
//                     tmp = dajpi[n][0].transpose() * db.col(n);
//                     term2 = term2 + tmp;
//                 }
//                 omega_pa.col(i) = X * (term1-(0.5 * darboux_vector[i] * length) * term2);
//                 //omega_pa.col(i) *= bendAndTwistKs[i];
//             }
//             // pb
//             {
//                 VECTOR3d term1(0, 0, 0);
//                 VECTOR3d term2(0, 0, 0);
//                 VECTOR3d tmp(0, 0, 0);
//                 // first term
//                 term1 = dajpi[j][1].transpose() * db.col(k);
//                 tmp =   dajpi[k][1].transpose() * db.col(j);
//                 term1 = term1 - tmp;
//                 // third term
//                 tmp = dbjpi[j][0].transpose() * da.col(k);
//                 term1 = term1 - tmp;
                
//                 tmp = dbjpi[k][0].transpose() * da.col(j);
//                 term1 = term1 + tmp;

//                 // second term
//                 for (int n = 0; n < 3; ++n) 
//                 {
//                     tmp = dajpi[n][1].transpose() * db.col(n);
//                     term2 = term2 + tmp;
                    
//                     tmp = dbjpi[n][0].transpose() * da.col(n);
//                     term2 = term2 + tmp;
//                 }
//                 omega_pb.col(i) = X * (term1-(0.5 * darboux_vector[i] * length) * term2);
//                 //omega_pb.col(i) *= bendAndTwistKs[i];
//             }
//             // pc
//             {
//                 VECTOR3d term1(0, 0, 0);
//                 VECTOR3d term2(0, 0, 0);
//                 VECTOR3d tmp(0, 0, 0);
                
//                 // first term
//                 term1 = dbjpi[j][1].transpose() * da.col(k);
//                 tmp =   dbjpi[k][1].transpose() * da.col(j);
//                 term1 = term1 - tmp;

//                 // second term
//                 for (int n = 0; n < 3; ++n) 
//                 {
//                     tmp = dbjpi[n][1].transpose() * da.col(n);
//                     term2 = term2 + tmp;
//                 }
//                 omega_pc.col(i) = -X*(term1+(0.5 * darboux_vector[i] * length) * term2);
//                 //omega_pc.col(i) *= bendAndTwistKs[i];
//             }
//             // pd
//             {
//                 VECTOR3d term1(0, 0, 0);
//                 VECTOR3d term2(0, 0, 0);
//                 VECTOR3d tmp(0, 0, 0);
//                 // first term
//                 term1 = dajpi[j][2].transpose() * db.col(k);
//                 tmp =   dajpi[k][2].transpose() * db.col(j);
//                 term1 = term1 - tmp;
//                 // second term
//                 for (int n = 0; n < 3; ++n) {
//                     tmp = dajpi[n][2].transpose() * db.col(n);
//                     term2 = term2 + tmp;
//                 }
//                 omega_pd.col(i) = X*(term1-(0.5 * darboux_vector[i] * length) * term2);
//                 //omega_pd.col(i) *= bendAndTwistKs[i];
//             }
//             // pe
//             {
//                 VECTOR3d term1(0, 0, 0);
//                 VECTOR3d term2(0, 0, 0);
//                 VECTOR3d tmp(0, 0, 0);
//                 // first term
//                 term1 = dbjpi[j][2].transpose() * da.col(k);
//                 tmp = dbjpi[k][2].transpose() * da.col(j);
//                 term1 -= tmp;
                
//                 // second term
//                 for (int n = 0; n < 3; ++n) 
//                 {	
//                     tmp = dbjpi[n][2].transpose() * da.col(n);
//                     term2 += tmp;
//                 }

//                 omega_pe.col(i) = -X*(term1+(0.5 * darboux_vector[i] * length) * term2);
//                 //omega_pe.col(i) *= bendAndTwistKs[i];
//             }
//         }
//         return true;
//     }

};
};