#include <example_based_quasi_static_solver.h>

int ExamBasedQuasiStaticSolver::EvalElmObj(const TetAttributes& attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psi;
        ComputeDeformationGradient(attrs._Minv,u0,F);
        force_model->ComputePsi(attrs,F,psi);

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = (psi * vol - u0.dot(gravity_force));

        Vec12d example_diff = u0 - attrs._example_pos;
        FEM_Scaler exam_energy = 0.5 * example_diff.transpose() * attrs._example_pos_weight.asDiagonal() * example_diff;
        *elm_obj += exam_energy;

        // *elm_obj = exam_energy;
        return 0;
}

int ExamBasedQuasiStaticSolver::EvalElmObjDeriv(const TetAttributes& attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psi;
        Vec9d dpsi;
        ComputeDeformationGradient(attrs._Minv,u0,F);
        force_model->ComputePsiDeriv(attrs,F,psi,dpsi);

        const Mat9x12d& dFdX = attrs._dFdX;

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = (psi * vol - u0.dot(gravity_force));
        elm_deriv = (vol * dFdX.transpose() * dpsi - gravity_force);

        Vec12d example_diff = u0 - attrs._example_pos;
        FEM_Scaler exam_energy = 0.5 * example_diff.transpose() * attrs._example_pos_weight.asDiagonal() * example_diff;
        *elm_obj += exam_energy;
        elm_deriv += attrs._example_pos_weight.asDiagonal() * example_diff;

        // *elm_obj = exam_energy;
        // elm_deriv = attrs._example_pos_weight.asDiagonal() * example_diff;
        return 0;
}

int ExamBasedQuasiStaticSolver::EvalElmObjDerivJacobi(const TetAttributes& attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv,Mat12x12d& elm_H,bool spd,bool debug) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        Vec9d dpsiE;
        Mat9x9d ddpsiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);


    if(debug){
        // std::cout << "DEBUG FOR QUASI_STATIC SOLVER" << std::endl;
        Mat3x3d Ftest = F;

        Mat9x9d ddpsi_cmp = Mat9x9d::Zero();
        Vec9d dpsi_cmp = Vec9d::Zero();
        FEM_Scaler psi_cmp = 0;

        force_model->ComputePsiDerivHessian(attrs,Ftest,psi_cmp,dpsi_cmp,ddpsi_cmp,false);

        Mat9x9d ddpsi_fd = Mat9x9d::Zero();
        Vec9d dpsi_fd = Vec9d::Zero();
        FEM_Scaler ratio = 1e-8;
        for(size_t i = 0;i < 9;++i){
                Mat3x3d F_tmp = Ftest;
                Vec9d f_tmp = MatHelper::VEC(F_tmp);
                FEM_Scaler step = f_tmp[i] * ratio;
                step = fabs(step) < ratio ? ratio : step;
                f_tmp[i] += step;
                F_tmp = MatHelper::MAT(f_tmp);

                FEM_Scaler psi_tmp;
                Vec9d dpsi_tmp;
                force_model->ComputePsiDeriv(attrs,F_tmp,psi_tmp,dpsi_tmp);

                dpsi_fd[i] = (psi_tmp - psi_cmp) / step;
                ddpsi_fd.col(i) = (dpsi_tmp - dpsi_cmp) / step;
        }

        FEM_Scaler ddpsi_error = (ddpsi_fd - ddpsi_cmp).norm() / ddpsi_fd.norm();
        FEM_Scaler dpsi_error = (dpsi_fd - dpsi_cmp).norm() / dpsi_fd.norm();
        if(ddpsi_error > 1e-3 || dpsi_error > 1e-3){
                std::cout << "TEST PLATIC" << std::endl << Ftest << std::endl;
                std::cerr << "dpsi_error : " << dpsi_error << std::endl;
                std::cerr << "dpsi_fd : \t" << dpsi_fd.transpose() << std::endl;
                std::cerr << "dpsi : \t" << dpsi_cmp.transpose() << std::endl; 
                std::cerr << "ddpsi_error : " << ddpsi_error << std::endl;
                std::cout << "ddpsi : " << std::endl << ddpsi_cmp << std::endl;
                std::cout << "ddpsi_fd : " << std::endl << ddpsi_fd << std::endl;

                std::cout << "pstrain : " << attrs.pmp.plastic_strain.transpose() << std::endl;
                std::cout << "kinimatic_hardening_shift : " << attrs.pmp.kinematic_hardening_shift.transpose() << std::endl;
                std::cout << "Ftest : " << std::endl << Ftest << std::endl;

                throw std::runtime_error("ddpsi_error");
        }
    }

    force_model->ComputePsiDerivHessian(attrs,F,psiE,dpsiE,ddpsiE,spd);

    if(std::isnan(F.norm()) || std::isnan(psiE) || std::isnan(dpsiE.norm()) || std::isnan(ddpsiE.norm())){
        std::cout << "F : " << std::endl << F << std::endl;
        std::cout << "psiE : " << psiE << std::endl;
        std::cout << "dpsiE : " << std::endl << dpsiE.transpose() << std::endl;
        std::cout << "ddpsiE : " << std::endl << ddpsiE << std::endl;
    }

    const Mat9x12d& dFdX = attrs._dFdX;

    Vec12d gravity_force = _gravity.replicate(4,1) * m;
    *elm_obj = (psiE * vol - u0.dot(gravity_force));
    elm_deriv = (vol * dFdX.transpose() * dpsiE - gravity_force);
    elm_H = (vol * dFdX.transpose() * ddpsiE * dFdX);

    if(std::isnan(elm_deriv.norm()) || std::isnan(elm_H.norm())){
        std::cout << "vol : " << vol << std::endl;
        std::cout << "dFdX : " << std::endl << dFdX << std::endl;
    }


    Vec12d example_diff = u0 - attrs._example_pos;
    FEM_Scaler exam_energy = 0.5 * example_diff.transpose() * attrs._example_pos_weight.asDiagonal() * example_diff;
    *elm_obj += exam_energy;
    elm_deriv += attrs._example_pos_weight.asDiagonal() * example_diff;
    elm_H.diagonal() += attrs._example_pos_weight;

    // *elm_obj = exam_energy;
    // elm_deriv = attrs._example_pos_weight.asDiagonal() * example_diff;
    // elm_H.setConstant(0);
    // elm_H.diagonal() = attrs._example_pos_weight;

    return 0;
}