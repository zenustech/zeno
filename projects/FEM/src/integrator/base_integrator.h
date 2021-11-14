#pragma once

#include "base_force_model.h"
#include "base_elastic_model.h"
#include "plastic_force_model.h"
#include "diriclet_damping.h"
#include <string>

// the TetAttributes encapsulate the information of trajectory-dependent information of force model
struct TetAttributes{
    size_t _elmID;
    Mat4x4d _Minv;
    Mat9x12d _dFdX;

    FEM_Scaler _volume;
    FEM_Scaler _density;
    Vec12d _ext_f;

    ElastoMaterialParam emp;
    PlasticMaterialParam pmp;// trajectory dependent force model
    FEM_Scaler v;
};

/**
 * @class <BaseIntegrator>
 * @brief A base integrator class for inheritance, providing the basic interface for override. The derived integrator should be defined by an optimization problem.
 * As we use Newont-Raphson or CG algorithm to solve the stepping, An interface providing objective function evaluation,derivative,Hessian and differentials evaluation is needed.
 */
class BaseIntegrator{
public:
    BaseIntegrator(size_t cou_length) {
        _couplingLength = cou_length;
        _gravity.setZero();
    }
    /**
     * @brief default destructor for base integrator.
     */
    virtual ~BaseIntegrator(void) {}
    /**
     * @return Get the coupling length of the integrator.
     */
    size_t GetCouplingLength() const {return _couplingLength;}    // the number succesive frames need to compute the residual at specific frame 
    /**
     * @brief set the time step
     */
    inline void SetTimeStep(FEM_Scaler dt) { _dt = dt;}
    /**
     * @return the time step.
     */
    inline FEM_Scaler GetTimeStep(void) const {return _dt;}
    /**
     * @brief Set the gravity acceleration.
     */
    inline void SetGravity(const Vec3d &gravity) { _gravity = gravity; } // set the gravitational acceleration
    /**
     * @brief Get the gravity acceleration.
     */
    inline const Vec3d& GetGravity() {return _gravity;}

public:
    /**
     * @brief Interface for defining element-wise objective function.
     * @param elm_id the element ID
     * @param _elm_states the successive element's states defining the objective function, _elm_states.size() == coupling_length, with _elm_states[coupling_length - 1] the current frame,
     * and _elm_states[coupling_length - 2] the previous frame ane so on. 
     * @param obj the objective function
     */
    virtual int EvalElmObj(const TetAttributes tet_attribs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj) const = 0;
    /**
     * @brief Interface for defining element-wise objective function and its derivative.
     * @param elm_id the element ID
     * @param _elm_states the successive element's states defining the objective function, _elm_states.size() == coupling_length, with _elm_states[coupling_length - 1] the current frame,
     * and _elm_states[coupling_length - 2] the previous frame ane so on.
     * @param obj the objective function
     * @param elm_deriv the derivative of objective function with respect to the element shape.
     */
    virtual int EvalElmObjDeriv(const TetAttributes tet_attribs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj,Vec12d& elm_deriv) const = 0;
    /**
     * @brief Interface for defining element-wise objective function, its derivative and Jacobi.
     * @param elm_id the element ID
     * @param _elm_states the successive element's states defining the objective function, _elm_states.size() == coupling_length, with _elm_states[coupling_length - 1] the current frame,
     * and _elm_states[coupling_length - 2] the previous frame ane so on.
     * @param obj the objective function
     * @param elm_deriv the derivative of objective function w.r.t the current element shape.
     * @param elm_H the Jacobi of the objective function's derivative w.r.t the element frame specified by coulping ID.
     * @param cou_id the coupling ID
     * @param enforce_spd whether enforcing the SPD of Jacobi matrix
     */
    virtual int EvalElmObjDerivJacobi(const TetAttributes tet_attribs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::shared_ptr<DiricletDampingModel>& damping_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool enforce_spd) const = 0;

    // virtual int EvalElmResDifferentials(const TetAttributes tet_attribs,
    //         const std::shared_ptr<BaseForceModel>& force_model,
    //         const std::shared_ptr<DiricletDampingModel>& damping_model,
    //         const std::vector<Vec12d>& elm_states,const Vec12d& dx,Vec12d& diff,bool enforce_spd) const = 0;


    static void ComputeDeformationGradient(const Mat4x4d& Minv,const Vec12d &elm_u,Mat3x3d& F) {
        Mat4x4d G;
        for(size_t i = 0;i < 4;++i)
            G.block(0,i,3,1) = elm_u.segment(i*3,3);
        G.bottomRows(1).setConstant(1.0);
        G = G * Minv;
        F = G.topLeftCorner(3,3);
    }

protected:
    size_t _couplingLength;
    Vec3d _gravity;
    FEM_Scaler _dt;
    FEM_Scaler _epsilon;

};
