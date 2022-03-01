#include <skin_driven_quasi_static_solver.h>

int SkinDrivenQuasiStaticSolver::EvalElmObj(const TetAttributes& attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj) const {
//    std::cout << "EVAL SKIN Driven OBJ" << std::endl;

        Vec12d u0 = elm_states[0];
        FEM_Scaler vol = attrs._volume;
        FEM_Scaler m = vol * attrs._density / 4;

        Mat3x3d F;
        FEM_Scaler psi;
        ComputeDeformationGradient(attrs._Minv,u0,F);
        force_model->ComputePsi(attrs,F,psi);

        Vec12d gravity_force = _gravity.replicate(4,1) * m;
        *elm_obj = (psi * vol - u0.dot(gravity_force));


        if(attrs.interpPs.size() > 0){
            for(size_t i = 0;i < attrs.interpPs.size();++i){
                const auto& ipos = attrs.interpPs[i];
                const auto& w = attrs.interpWs[i];
                Vec4d iw;iw << w[0],w[1],w[2],1-w[0]-w[1]-w[2];

                Vec3d tpos = Vec3d::Zero();
                for(size_t j = 0;j < 4;++j)
                    tpos += iw[j] * u0.segment(j*3,3);

                *elm_obj += 0.5 * attrs.interpPenaltyCoeff * (tpos - ipos).squaredNorm() / attrs.interpPs.size();
            }
        }

        return 0;
}

int SkinDrivenQuasiStaticSolver::EvalElmObjDeriv(const TetAttributes& attrs,
    const std::shared_ptr<BaseForceModel>& force_model,
    const std::shared_ptr<DiricletDampingModel>& damping_model,
    const std::vector<Vec12d>& elm_states,FEM_Scaler* elm_obj,Vec12d& elm_deriv) const {

//    std::cout << "EVAL SKIN Driven Derivative" << std::endl;

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

        if(attrs.interpPs.size() > 0){
            for(size_t i = 0;i < attrs.interpPs.size();++i){
                const auto& ipos = attrs.interpPs[i];
                const auto& w = attrs.interpWs[i];
                Vec4d iw;
                iw << w[0],w[1],w[2],1-w[0]-w[1]-w[2];

                Vec3d tpos = Vec3d::Zero();
                for(size_t j = 0;j < 4;++j)
                    tpos += iw[j] * u0.segment(j*3,3);

                *elm_obj += 0.5 * attrs.interpPenaltyCoeff * (tpos - ipos).squaredNorm() / attrs.interpPs.size();

                for(size_t j = 0;j < 4;++j)
                    elm_deriv.segment(j*3,3) += attrs.interpPenaltyCoeff * iw[j] * (tpos - ipos) / attrs.interpPs.size();
            }
        }


        return 0;
}

int SkinDrivenQuasiStaticSolver::EvalElmObjDerivJacobi(const TetAttributes& attrs,
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


//    std::cout << "EVAL SKIN Driven Jacobi" << std::endl;

    // debug = true;
    // if(debug){
    //     // std::cout << "DEBUG FOR QUASI_STATIC SOLVER" << std::endl;
    //     Mat3x3d Ftest = F;

    //     Mat9x9d ddpsi_cmp = Mat9x9d::Zero();
    //     Vec9d dpsi_cmp = Vec9d::Zero();
    //     FEM_Scaler psi_cmp = 0;

    //     force_model->ComputePsiDerivHessian(attrs,Ftest,psi_cmp,dpsi_cmp,ddpsi_cmp,false);

    //     Mat9x9d ddpsi_fd = Mat9x9d::Zero();
    //     Vec9d dpsi_fd = Vec9d::Zero();
    //     FEM_Scaler ratio = 1e-8;
    //     for(size_t i = 0;i < 9;++i){
    //             Mat3x3d F_tmp = Ftest;
    //             Vec9d f_tmp = MatHelper::VEC(F_tmp);
    //             FEM_Scaler step = f_tmp[i] * ratio;
    //             step = fabs(step) < ratio ? ratio : step;
    //             f_tmp[i] += step;
    //             F_tmp = MatHelper::MAT(f_tmp);

    //             FEM_Scaler psi_tmp;
    //             Vec9d dpsi_tmp;
    //             force_model->ComputePsiDeriv(attrs,F_tmp,psi_tmp,dpsi_tmp);

    //             dpsi_fd[i] = (psi_tmp - psi_cmp) / step;
    //             ddpsi_fd.col(i) = (dpsi_tmp - dpsi_cmp) / step;
    //     }

    //     FEM_Scaler ddpsi_error = (ddpsi_fd - ddpsi_cmp).norm() / ddpsi_fd.norm();
    //     FEM_Scaler dpsi_error = (dpsi_fd - dpsi_cmp).norm() / dpsi_fd.norm();
    //     if((ddpsi_error > 1e-3 || dpsi_error > 1e-3) && attrs.interpPs.size() > 0){
    //             // std::cout << "TEST PLATIC" << std::endl << Ftest << std::endl;
    //             std::cout << "ELM_ID : " << attrs._elmID << std::endl;
    //             std::cout << "NM_INTERP_P : " << attrs.interpPs.size() << std::endl;
    //             std::cerr << "dpsi_error : " << dpsi_error << std::endl;
    //             std::cerr << "dpsi_fd : \t" << dpsi_fd.norm() << std::endl;
    //             std::cerr << "dpsi : \t" << dpsi_cmp.norm() << std::endl; 
    //             // std::cerr << "ddpsi_error : " << ddpsi_error << std::endl;
    //             // std::cout << "ddpsi : " << std::endl << ddpsi_cmp << std::endl;
    //             // std::cout << "ddpsi_fd : " << std::endl << ddpsi_fd << std::endl;

    //             std::cout << "pstrain : " << attrs.pmp.plastic_strain.transpose() << std::endl;
    //             std::cout << "kinimatic_hardening_shift : " << attrs.pmp.kinematic_hardening_shift.transpose() << std::endl;
    //             std::cout << "Ftest : " << std::endl << Ftest << std::endl;

    //             std::cout << "InterpCoeff : \n" << attrs.interpPenaltyCoeff << std::endl;

    //             for(size_t i = 0;i < attrs.interpPs.size();++i)
    //                 std::cout << "V : " << attrs.interpPs[i].transpose() << std::endl \
    //                     << "W : " << attrs.interpWs[i].transpose() << std::endl;

    //             throw std::runtime_error("ddpsi_error");
    //     }
    // }

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


    if(attrs.interpPs.size() > 0){
        for(size_t i = 0;i < attrs.interpPs.size();++i){
            const auto& ipos = attrs.interpPs[i];
            const auto& w = attrs.interpWs[i];
            Vec4d iw;
            iw << w[0],w[1],w[2],1-w[0]-w[1]-w[2];

            Vec3d tpos = Vec3d::Zero();
            for(size_t j = 0;j < 4;++j)
                tpos += iw[j] * u0.segment(j*3,3);

            *elm_obj += 0.5 * attrs.interpPenaltyCoeff * (tpos - ipos).squaredNorm() / attrs.interpPs.size();

            Vec12d pos_diff;

            for(size_t j = 0;j < 4;++j){
                elm_deriv.segment(j*3,3) += attrs.interpPenaltyCoeff * iw[j] * (tpos - ipos) / attrs.interpPs.size();
                pos_diff.segment(j*3,3) = tpos - ipos;
            }

            // std::cout << "POS_DIFF : " << pos_diff.transpose() << "\t" << attrs.interpPenaltyCoeff * iw.transpose() << std::endl;

            for(size_t j = 0;j < 4;++j)
                for(size_t k = 0;k < 4;++k){
                    FEM_Scaler alpha = attrs.interpPenaltyCoeff * iw[j] * iw[k] / attrs.interpPs.size();
                    elm_H.block(j * 3,k*3,3,3).diagonal() += Vec3d::Constant(alpha);
                }
        }
    }


    return 0;
}