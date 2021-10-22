#pragma once

#include <memory>

#include <differentiable_SVD.h>

#include <iostream>
#include <limits>

/**
 * @class <BaseForceModel>
 * @brief A force model base class for inheritance, which defines the common interfaces needed for defining all the force model derived classes 
 * 
 * The current version only support isotropic elasto and damping model. The force model base class is unaware of the TetMesh, the input to all its
 * function is deformation gradient of individual element, instead of its deformaed shape.
 */
class BaseForceModel {
public:
    /**
     * @brief The constructor of BaseForceModel class
     */
    BaseForceModel() {}
    /**
     * @brief destructor method.
     */
    virtual ~BaseForceModel(){}
    /**
     * @brief An interface for defining the potential energy of anisotropic force model, all the force models should inherit this method and implement their 
     * own version of element-wise potential energy defination.
     * @param activation the activation level along the fiber direction
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     */
    virtual void ComputePhi(const Mat3x3d& activation,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d& F,FEM_Scaler& energy) const = 0;
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination and element-wise energy gradient.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    virtual void ComputePhiDeriv(const Mat3x3d& activation,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d& F,FEM_Scaler &energy,Vec9d &derivative) const = 0;
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination, element-wise energy gradient and 12x12 element-wise energy hessian w.r.t deformed shape.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param fiber_direction the three orthogonal fiber directions
     * @param <F> the deformation gradient
     * @param <energy> the potential energy output
     * @param <derivative> the derivative of potential energy w.r.t the deformation gradient
     * @param <Hessian> the hessian of potential energy w.r.t the deformed shape for elasto model or nodal velocities for damping model
     * @param <enforcing_spd> decide whether we should enforce the SPD of hessian matrix
     */
    virtual void ComputePhiDerivHessian(const Mat3x3d& activation,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d &F,FEM_Scaler& energy,Vec9d &derivative, Mat9x9d &Hessian,bool enforcing_spd = true) const = 0;

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
// we place all the invarients, their derivs and Hessians computation here
};
