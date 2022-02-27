#pragma once

#include <memory>

#include <differentiable_SVD.h>

#include <iostream>
#include <limits>

struct ElastoMaterialParam {
    FEM_Scaler E;
    FEM_Scaler nu;
    Mat3x3d Act;
    Vec3d forient;
};


struct PlasticMaterialParam{
    // ElastoMaterialParam emp;

    FEM_Scaler yield_stress;
    Vec3d init_stress;
    Vec3d init_strain;

    FEM_Scaler kinematic_hardening_coeff;
    Vec3d kinematic_hardening_shift;

    FEM_Scaler isotropic_hardening_coeff;
    
    Vec3d plastic_strain;
    Mat3x3d PS;

    FEM_Scaler restoring_strain;
    FEM_Scaler failed_strain;

    bool failed;
    Vec3d the_strain_failed;// the strain at the very first time the material break down
    Mat3x3d F_failed;
};
struct TetAttributes{
    size_t _elmID;
    Mat4x4d _Minv;
    Mat9x12d _dFdX;

    FEM_Scaler _volume;
    FEM_Scaler _density;
    Vec12d _ext_f;

    // example-based energy term
    Vec12d _example_pos;
    Vec12d _example_pos_weight;

    ElastoMaterialParam emp;
    PlasticMaterialParam pmp;// trajectory dependent force model
    FEM_Scaler v;

    // interpolation energy form
    std::vector<Vec3d> interpWs;
    std::vector<Vec3d> interpPs;
    FEM_Scaler interpPenaltyCoeff;
};

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
     * @param forient the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     */
    virtual void ComputePsi(const TetAttributes& attrs,const Mat3x3d& F,FEM_Scaler& psi) const = 0;
    // virtual void ComputePsi(const TetAttributes& attrs,const Vec3d& sigma,FEM_Scaler& psi) const = 0;

    // virtual void ComputePsi() const = 0;
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination and element-wise energy gradient.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param forient the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    virtual void ComputePsiDeriv(const TetAttributes& attrs,const Mat3x3d& F,FEM_Scaler &psi,Vec9d &dpsi) const = 0;
    // virtual void ComputePsiDeriv(const TetAttributes& attrs,const Vec3d& sigma,FEM_Scaler &psi,Vec3d &dpsi) const = 0;
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination, element-wise energy gradient and 12x12 element-wise energy hessian w.r.t deformed shape.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param forient the three orthogonal fiber directions
     * @param <F> the deformation gradient
     * @param <energy> the potential energy output
     * @param <derivative> the derivative of potential energy w.r.t the deformation gradient
     * @param <Hessian> the hessian of potential energy w.r.t the deformed shape for elasto model or nodal velocities for damping model
     * @param <spd> decide whether we should enforce the SPD of hessian matrix
     */
    virtual void ComputePsiDerivHessian(const TetAttributes& attrs,const Mat3x3d &F,FEM_Scaler& psi,Vec9d &dpsi, Mat9x9d &ddpsi,bool spd = true) const = 0;


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
};
