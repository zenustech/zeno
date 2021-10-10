#include "backward_euler_integrator.h"
#include <iostream>
#include <ctime>
#include <numeric>

int BackEulerIntegrator::EvalElmObj(const TetAttributes attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
            
    FEM_Scaler m = attrs._volume * attrs._density / 4;  
    Vec12d v2 = (u2 - u1)/_dt;
   
    FEM_Scaler E_phi,I_phi;

    Mat3x3d Fu;
    ComputeDeformationGradient(attrs._Minv,u2,Fu);
    force_model->ComputePhi(attrs._activation,
        attrs._fiberWeight,
        attrs._fiberOrient,
        attrs._E,
        attrs._nu,
        Fu,
        E_phi);       

    Vec12d y = u2 - 2*u1 + u0 - _dt*_dt*_gravity.replicate(4,1); 
    I_phi = y.squaredNorm() * m / 2 / _dt;

    *elm_obj = I_phi + _dt*E_phi*attrs._volume;

    return 0;
}

int BackEulerIntegrator::EvalElmObjDeriv(const TetAttributes attrs,
            const std::shared_ptr<BaseForceModel>& force_model,
            const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
          
    FEM_Scaler m = attrs._volume * attrs._density / 4;  
    Vec12d v2 = (u2 - u1)/_dt;

    Mat3x3d Fu;
    ComputeDeformationGradient(attrs._Minv,u2,Fu);
    FEM_Scaler E_phi,I_phi;
    Vec9d E_deriv;

    force_model->ComputePhiDeriv(attrs._activation,
        attrs._fiberWeight,
        attrs._fiberOrient,
        attrs._E,
        attrs._nu,
        Fu,
        E_phi,E_deriv);

    // {
    //     FEM_Scaler phi_tmp;
    //     Vec9d deriv_fd;
    //     Mat3x3d Fu_tmp = Fu;
    //     for(size_t i = 0;i < 9;++i){
    //         Fu_tmp = Fu;
    //         Vec9d Fu_vec = MatHelper::VEC(Fu);
    //         FEM_Scaler step = Fu_vec[i] * 1e-7;
    //         step = fabs(step) < 1e-7 ? 1e-7 : step;
    //         Fu_vec[i] += step;
    //         Fu_tmp = MatHelper::MAT(Fu_vec);

    //         force_model->ComputePhi(attrs._activation,
    //             attrs._fiberWeight,
    //             attrs._fiberOrient,
    //             attrs._E,
    //             attrs._nu,
    //             Fu_tmp,
    //             phi_tmp);
    //         deriv_fd[i] = (phi_tmp - E_phi) / step;
    //     }

    //     FEM_Scaler D_error = (deriv_fd - E_deriv).norm() / E_deriv.norm();
    //     if(D_error > 1e-4){
    //         std::cout << "ATTR: " << std::endl;
    //         std::cout << "ACT: " << std::endl << attrs._activation << std::endl;
    //         std::cout << "WEIGHT: " << std::endl << attrs._fiberWeight << std::endl;
    //         std::cout << "ORIENT: " << std::endl << attrs._fiberOrient << std::endl;
    //         std::cout << "ORIENT_DET : " << attrs._fiberOrient.determinant() << std::endl;
    //         std::cout << "E : " << attrs._E << std::endl;
    //         std::cout << "nu :" << attrs._nu << std::endl;
    //         std::cout << "Fu : " << std::endl << Fu << std::endl;

    //         std::cout << "FORCE_ELM_D_ERROR : " << D_error << std::endl;
    //         for(size_t i = 0;i < 9;++i)
    //             std::cout << "idx : " << i << "\t" << deriv_fd[i] << "\t" << E_deriv[i] << std::endl;
    //         throw std::runtime_error("FORCE_ELM_D_ERROR");
    //     }
    //     throw std::runtime_error("FORCE_ELM_D_CHECK");

        // std::cout << "force_model deriv pass" << std::endl;
    // }

    Vec12d y = u2 - 2*u1 + u0 - _dt*_dt*_gravity.replicate(4,1);
    I_phi = y.squaredNorm() * m / 2 / _dt;

    const Mat9x12d& dFdX = attrs._dFdX;

    *elm_obj = I_phi + _dt*E_phi * attrs._volume;   

    elm_deriv = m*y/_dt + _dt * dFdX.transpose() * E_deriv * attrs._volume;

    return 0;                
}

int BackEulerIntegrator::EvalElmObjDerivJacobi(const TetAttributes attrs,
        const std::shared_ptr<BaseForceModel>& force_model,
        const std::vector<Vec12d>& elm_states,
        FEM_Scaler* elm_obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool filtering) const{
    Vec12d u2 = elm_states[2];
    Vec12d u1 = elm_states[1];
    Vec12d u0 = elm_states[0];
         
    FEM_Scaler m = attrs._volume * attrs._density / 4;   
    Vec12d v2 = (u2 - u1)/_dt;

    Mat3x3d Fu;
    ComputeDeformationGradient(attrs._Minv,u2,Fu);

    FEM_Scaler E_phi,I_phi;
    Vec9d E_deriv;
    Mat9x9d E_Hessian;

    force_model->ComputePhiDerivHessian(attrs._activation,
        attrs._fiberWeight,
        attrs._fiberOrient,
        attrs._E,
        attrs._nu,
        Fu,
        E_phi,E_deriv,E_Hessian,filtering);

    FEM_Scaler vol = attrs._volume;


    Vec12d y = u2 - 2*u1 + u0 - _dt*_dt*_gravity.replicate(4,1);
    I_phi = y.squaredNorm() * m / 2 / _dt;
    *elm_obj = I_phi + _dt * E_phi * attrs._volume;   

    const Mat9x12d& dFdX = attrs._dFdX;
    elm_deriv = m * y / _dt + _dt * dFdX.transpose() * E_deriv * attrs._volume;

    // if(attrs._elmID == 0){
    //     std::cout << "residual_force : " << elm_deriv.norm() << std::endl;
    // }

    elm_H = Mat12x12d::Identity() * m / _dt + attrs._volume * dFdX.transpose() * (_dt * E_Hessian) * dFdX;

    return 0;
}