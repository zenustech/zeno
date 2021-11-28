#include <quasi_static_solver.h>

int QuasiStaticSolver::EvalElmObj(const TetAttributes attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);

        if(dynamic_cast<ElasticModel*>(force_model.get())){
            auto model = dynamic_cast<ElasticModel*>(force_model.get());
            model->ComputePhi(attrs.emp,F,psiE);

            if(attrs._elmID == 0){
                std::cout << "PSI0 = " << psiE << std::endl;
                std::cout << "PARAMS : " << attrs.emp.E << "\t" << attrs.emp.nu << "\t" << std::endl << attrs.emp.Act << std::endl << attrs.emp.forient.transpose() << std::endl;
            }
        }else{
            auto model = dynamic_cast<PlasticForceModel*>(force_model.get());
            model->ComputePsi(attrs.pmp,attrs.emp,F,psiE);
        }

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);

        return 0;
}

int QuasiStaticSolver::EvalElmObjDeriv(const TetAttributes attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {
        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psiE;
        Vec9d dpsiE;
        ComputeDeformationGradient(attrs._Minv,u0,F);

        if(dynamic_cast<ElasticModel*>(force_model.get())){
            auto model = dynamic_cast<ElasticModel*>(force_model.get());
            model->ComputePhiDeriv(attrs.emp,F,psiE,dpsiE);
        }else{
            auto model = dynamic_cast<PlasticForceModel*>(force_model.get());
            model->ComputePsiDeriv(attrs.pmp,attrs.emp,F,psiE,dpsiE);
        }
        const Mat9x12d& dFdX = attrs._dFdX;

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = psiE * vol - u0.dot(gravity_force);
        elm_deriv = vol * dFdX.transpose() * dpsiE - gravity_force;

        return 0;
}

int QuasiStaticSolver::EvalElmObjDerivJacobi(const TetAttributes attrs,
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

        dynamic_cast<PlasticForceModel*>(force_model.get())->ComputePsiDerivHessian(attrs.pmp,attrs.emp,Ftest,psi_cmp,dpsi_cmp,ddpsi_cmp,false);

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
                dynamic_cast<PlasticForceModel*>(force_model.get())->ComputePsiDeriv(attrs.pmp,attrs.emp,F_tmp,psi_tmp,dpsi_tmp);

                dpsi_fd[i] = (psi_tmp - psi_cmp) / step;
                ddpsi_fd.col(i) = (dpsi_tmp - dpsi_cmp) / step;
        }

        FEM_Scaler ddpsi_error = (ddpsi_fd - ddpsi_cmp).norm() / ddpsi_fd.norm();
        FEM_Scaler dpsi_error = (dpsi_fd - dpsi_cmp).norm() / dpsi_fd.norm();
        if(ddpsi_error > 1e-3 || dpsi_error > 1e-3){
                // Vec9d eig_vals;
                // Vec9d eig_vecs[9];

                // dynamic_cast<StableStvk*>(force_model.get())->ComputeIsoEigenSystem(lambda,mu,Ftest,eig_vals,eig_vecs);
                // std::cout << "Check Eigen !!!!!: " << std::endl;
                // for(size_t i = 0;i < 9;++i){
                //     Vec9d Ax = ddpsi_fd * eig_vecs[i];
                //     Vec9d lx = eig_vals[i] * eig_vecs[i];
                //     std::cout << "IDX : " << i << std::endl;
                //     for(size_t j = 0;j < 9;++j){
                //         std::cout << Ax[j] /  lx[j] << "\t";std::cout.flush();
                //     }
                //     std::cout << std::endl;
                // }
                std::cout << "TEST PLATIC" << std::endl << Ftest << std::endl;
                std::cerr << "dpsi_error : " << dpsi_error << std::endl;
                std::cerr << "dpsi_fd : \t" << dpsi_fd.transpose() << std::endl;
                std::cerr << "dpsi : \t" << dpsi_cmp.transpose() << std::endl; 
                std::cerr << "ddpsi_error : " << ddpsi_error << std::endl;
                std::cout << "ddpsi : " << std::endl << ddpsi_cmp << std::endl;
                std::cout << "ddpsi_fd : " << std::endl << ddpsi_fd << std::endl;

                std::cout << "pstrain : " << attrs.pmp.plastic_strain.transpose() << std::endl;
                std::cout << "kinimatic_hardening_shift : " << attrs.pmp.kinematic_hardening_shift.transpose() << std::endl;


                // throw std::runtime_error("ddpsi_error");
        }
    }


    if(dynamic_cast<ElasticModel*>(force_model.get())){
        auto model = dynamic_cast<ElasticModel*>(force_model.get());
        model->ComputePhiDerivHessian(attrs.emp,F,psiE,dpsiE,ddpsiE,spd);
    }else{
        auto model = dynamic_cast<PlasticForceModel*>(force_model.get());
        model->ComputePsiDerivHessian(attrs.pmp,attrs.emp,F,psiE,dpsiE,ddpsiE,spd);
    }

    const Mat9x12d& dFdX = attrs._dFdX;

    Vec12d gravity_force = _gravity.replicate(4,1) * m;
    *elm_obj = psiE * vol - u0.dot(gravity_force);
    elm_deriv = vol * dFdX.transpose() * dpsiE - gravity_force;

    elm_H = vol * dFdX.transpose() * ddpsiE * dFdX;

    // std::cout << "OUT : " << std::endl << ddpsiE << std::endl;

    return 0;
}