#include <quasi_static_solver.h>

int QuasiStaticSolver::EvalElmObj(const TetAttributes& attrs,
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

        Vec12d vol_force = attrs._volume_force_accel.replicate(4,1) * m;
        Vec12d nodal_force = attrs._nodal_force;
        *elm_obj = (psi * vol - u0.dot(vol_force + nodal_force));

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
        Vec12d example_diff = u0 - attrs._example_pos;
        FEM_Scaler exam_energy = 0.5 * example_diff.transpose() * attrs._example_pos_weight.asDiagonal() * example_diff;
        *elm_obj += exam_energy;

        return 0;
}

int QuasiStaticSolver::EvalElmObjDeriv(const TetAttributes& attrs,
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

        Vec12d vol_force = attrs._volume_force_accel.replicate(4,1) * m;
        Vec12d nodal_force = attrs._nodal_force;
        *elm_obj = (psi * vol - u0.dot(vol_force + nodal_force));
        elm_deriv = (vol * dFdX.transpose() * dpsi - (vol_force + nodal_force));

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


        Vec12d example_diff = u0 - attrs._example_pos;
        FEM_Scaler exam_energy = 0.5 * example_diff.transpose() * attrs._example_pos_weight.asDiagonal() * example_diff;
        *elm_obj += exam_energy;
        elm_deriv += attrs._example_pos_weight.asDiagonal() * example_diff;

        return 0;
}

int QuasiStaticSolver::EvalElmObjDerivJacobi(const TetAttributes& attrs,
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

    force_model->ComputePsiDerivHessian(attrs,F,psiE,dpsiE,ddpsiE,spd);

    const Mat9x12d& dFdX = attrs._dFdX;

    Vec12d vol_force = attrs._volume_force_accel.replicate(4,1) * m;
    Vec12d nodal_force = attrs._nodal_force;
    *elm_obj = (psiE * vol - u0.dot(vol_force + nodal_force));
    elm_deriv = (vol * dFdX.transpose() * dpsiE - (vol_force + nodal_force));
    elm_H = (vol * dFdX.transpose() * ddpsiE * dFdX);


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

    Vec12d example_diff = u0 - attrs._example_pos;
    FEM_Scaler exam_energy = 0.5 * example_diff.transpose() * attrs._example_pos_weight.asDiagonal() * example_diff;
    *elm_obj += exam_energy;
    elm_deriv += attrs._example_pos_weight.asDiagonal() * example_diff;
    elm_H.diagonal() += attrs._example_pos_weight;

    return 0;
}