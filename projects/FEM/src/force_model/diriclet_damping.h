#pragma once

#include <memory>

#include <iostream>
#include <limits>

#include <matrix_helper.hpp>

/**
 * @class <DiricletDampingModel>
 * @brief A force model base class for inheritance, which defines the common interfaces needed for defining all the force model derived classes 
 * 
 * The current version only support isotropic elasto and damping model. The force model base class is unaware of the TetMesh, the input to all its
 * function is deformation gradient of individual element, instead of its deformaed shape.
 */
class DiricletDampingModel {
public:
    /**
     * @brief The constructor of DiricletDampingModel class
     */
    DiricletDampingModel() {}
    /**
     * @brief destructor method.
     */
    ~DiricletDampingModel(){}
    /**
     * @brief An interface for defining the potential energy of anisotropic force model, all the force models should inherit this method and implement their 
     * own version of element-wise potential energy defination.
     * @param fiber_direction the three orthogonal fiber directions
     * @param L the deformation gradient
     * @param energy the potential energy output
     */
    void ComputePhi(const FEM_Scaler& v,const Mat3x3d& L,FEM_Scaler& phi) const {
        phi = L.squaredNorm() * v / 2;
    }
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination and element-wise energy gradient.
     * @param fiber_direction the three orthogonal fiber directions
     * @param L the deformation gradient
     * @param energy the potential energy output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    virtual void ComputePhiDeriv(const FEM_Scaler& v,const Mat3x3d& L,FEM_Scaler &phi,Vec9d &deriv) const {
        ComputePhi(v,L,phi);
        deriv = v * MatHelper::VEC(L);
    }
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
    void ComputePhiDerivHessian(const FEM_Scaler& v,const Mat3x3d& L,FEM_Scaler& phi,Vec9d &deriv, Mat9x9d &H) const {
        ComputePhiDeriv(v,L,phi,deriv);
        H = v * Mat9x9d::Identity();
    }
};
