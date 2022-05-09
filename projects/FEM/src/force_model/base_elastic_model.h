#pragma once

#include <memory>

#include <differentiable_SVD.h>
#include <base_force_model.h>

#include <iostream>
#include <limits>
#include <array>

/**
 * @class <ElasticModel>
 * @brief A force model base class for inheritance, which defines the common interfaces needed for defining all the force model derived classes 
 * 
 * The current version only support isotropic elasto and damping model. The force model base class is unaware of the TetMesh, the input to all its
 * function is deformation gradient of individual element, instead of its deformaed shape.
 */

class ElasticModel : public BaseForceModel {
public:
    /**
     * @brief The constructor of ElasticModel class
     */
    ElasticModel() : BaseForceModel() {
        Qs[0] <<   1, 0, 0,
                0, 0, 0,
                0, 0, 0;
        Qs[1]  <<   0, 0, 0,
                0, 1, 0,
                0, 0, 0;
        Qs[2]  <<   0, 0, 0,
                0, 0, 0,
                0, 0, 1;

        Qs[3]  <<   0,-1, 0,
                1, 0, 0,
                0, 0, 0;
        Qs[3]  /= sqrt(2);
    
        Qs[4]  <<   0, 0, 0,
                0, 0, 1,
                0,-1, 0;
        Qs[4]  /= sqrt(2);

        Qs[5]  <<   0, 0, 1,
                0, 0, 0,
                -1,0, 0;
        Qs[5]  /= sqrt(2);

        Qs[6]  <<   0, 1, 0,
                1, 0, 0,
                0, 0, 0;
        Qs[6]  /= sqrt(2);

        Qs[7]  <<   0, 0, 0,
                0, 0, 1,
                0, 1, 0;
        Qs[7]  /= sqrt(2);

        Qs[8]  <<   0, 0, 1,
                0, 0, 0,
                1, 0, 0;
        Qs[8]  /= sqrt(2);
    }
    /**
     * @brief destructor method.
     */
    virtual ~ElasticModel(){}
    // virtual void ComputePhiDerivHessian(const TetAttributes& attrs,const Vec3d &sigma,FEM_Scaler& psi,Vec3d &dpsi, Mat3x3d &ddpsi) const = 0;

    virtual void ComputePrincipalStress(const TetAttributes& attrs,const Vec3d& pstrain,Vec3d& pstress) const = 0;

    virtual void ComputePrincipalStressJacobi(const TetAttributes& attrs,const Vec3d& strain,Vec3d& stress,Mat3x3d& Jac) const = 0;

    static FEM_Scaler Enu2Lambda(FEM_Scaler E,FEM_Scaler nu) {return (nu * E)/((1 + nu) * (1 - 2*nu));}
    static FEM_Scaler Enu2Mu(FEM_Scaler E,FEM_Scaler nu) {return E / (2 * (1 + nu));}
    static FEM_Scaler Lame2E(FEM_Scaler lambda,FEM_Scaler mu) {return mu*(2*mu+3*lambda)/(mu+lambda);}
    static FEM_Scaler Lame2Nu(FEM_Scaler lambda,FEM_Scaler mu) {return lambda / (2 * (lambda +mu));}

    inline Mat9x9d EvaldFactdF(const Mat3x3d& Act_inv) const {
        Mat9x9d M = Mat9x9d::Zero();
        
        M(0,0) = M(1,1) = M(2,2) = Act_inv(0,0);
        M(3,0) = M(4,1) = M(5,2) = Act_inv(0,1);
        M(6,0) = M(7,1) = M(8,2) = Act_inv(0,2);

        M(0,3) = M(1,4) = M(2,5) = Act_inv(1,0);
        M(3,3) = M(4,4) = M(5,5) = Act_inv(1,1);
        M(6,3) = M(7,4) = M(8,5) = Act_inv(1,2);

        M(0,6) = M(1,7) = M(2,8) = Act_inv(2,0);
        M(3,6) = M(4,7) = M(5,8) = Act_inv(2,1);
        M(6,6) = M(7,7) = M(8,8) = Act_inv(2,2);

        return M;
    }

    static FEM_Scaler GetStiffness(const FEM_Scaler& E,const FEM_Scaler& nu,const FEM_Scaler& x_start,const FEM_Scaler& x_end){
        auto mu = Enu2Mu(E,nu);
        auto lambda = Enu2Lambda(E,nu);
        return mu + lambda;    
    }

    inline void EvalIsoInvarients(const Mat3x3d& F,Vec3d& Is) const{
        Mat3x3d U,V;
        Vec3d sigma;
        DiffSVD::SVD_Decomposition(F, U, sigma, V);        

        Is[0] = sigma.sum();
        Is[1] = sigma.squaredNorm();
        Is[2] = sigma[0] * sigma[1] * sigma[2];
    }

    inline void EvalIsoInvarients(const Vec3d& sigma,Vec3d& Is) const{
        Is[0] = sigma.sum();
        Is[1] = sigma.squaredNorm();
        Is[2] = sigma[0] * sigma[1] * sigma[2];
    }
    
    inline void EvalIsoInvarientsDeriv(const Mat3x3d& F,
            Vec3d& Is,
            std::array<Vec9d,3>& gs) const {
        Mat3x3d U,V;
        Vec3d sigma;
        DiffSVD::SVD_Decomposition(F, U, sigma, V);        

        Is[0] = sigma.sum();
        Is[1] = sigma.squaredNorm();
        Is[2] = sigma[0] * sigma[1] * sigma[2];

        Mat3x3d R = U * V.transpose();

        gs[0] = MatHelper::VEC(R);
        gs[1] = 2 * MatHelper::VEC(F);

        Mat3x3d J;
        J.col(0) = F.col(1).cross(F.col(2));
        J.col(1) = F.col(2).cross(F.col(0));
        J.col(2) = F.col(0).cross(F.col(1));       

        gs[2] = MatHelper::VEC(J);
    }

    inline void EvalAnisoInvarients(const Mat3x3d& F,const Vec3d& a,FEM_Scaler& Ia) const {
        Vec3d fa = F * a;
        Ia = fa.squaredNorm();
    }

    inline void EvalAnisoInvarientsDeriv(const Mat3x3d& F,const Vec3d& a,FEM_Scaler& Ia,Vec9d& ga) const {
        Vec3d fa = F * a;
        Ia = fa.squaredNorm();
        ga = 2 * MatHelper::VEC(F * MatHelper::DYADIC(a,a));
    }

protected:
    Mat3x3d Qs[9];
};
