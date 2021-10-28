#pragma once

#include <vector>
#include "differentiable_SVD.h"
#include "base_force_model.h"

#include <algorithm>

/**
 * @class <IsotropicForceModel>
 * @brief a base class for isotropic force model, including both elasto and visco force model
 */
class AnisotropicSNHModel : public MuscleForceModel{
public:
    /**
     * @brief the constructor for isotropic force model
     */
    AnisotropicSNHModel() : MuscleForceModel() {}
    /**
     * @brief the destructor for isotropic force model
     */
    ~AnisotropicSNHModel() {}

    void ComputePhi(const Mat3x3d& Act,
            const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
            const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
            const Mat3x3d& F,FEM_Scaler& phi) const override {
        Vec3d Is;
        Mat3x3d F_act = F * Act.inverse();
        ComputeAnisotrpicInvarients(fiber_direction,aniso_weight,F_act,Is);
        // std::cout << "ComputePhi Is " << Is.transpose() << "\n" << fiber_direction << "\n" << aniso_weight << "\n" << F_act << std::endl;
        FEM_Scaler I1_d = evalI1_delta(aniso_weight,fiber_direction,F_act);

        FEM_Scaler E = YoungModulus;
        FEM_Scaler nu = PossonRatio;
        FEM_Scaler lambda = Enu2Lambda(E,nu);
        FEM_Scaler mu = Enu2Mu(E,nu);

        phi = mu/2 * (Is[0] - I1_d) + lambda/2 * (Is[2] - 1) * (Is[2] - 1);
    }
    void ComputePhiDeriv(const Mat3x3d& Act,
            const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
            const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
            const Mat3x3d& F,FEM_Scaler &phi,Vec9d &dphi) const override{
        

        Vec3d Is;
        std::vector<Vec9d> Ds(3);

        Mat3x3d A_inv = Act.inverse();
        Mat3x3d F_act = F * A_inv;

        ComputeAnisotropicInvarientsDeriv(fiber_direction,aniso_weight,F_act,Is,Ds);
        // std::cout << "ComputePhiDeriv Is " << Is.transpose() << "\n" << fiber_direction << "\n" << aniso_weight << "\n" << F_act << std::endl;


        FEM_Scaler I1_d = evalI1_delta(aniso_weight,fiber_direction,F_act);
        Vec9d I1_d_deriv = MatHelper::VEC(evalI1_delta_deriv(aniso_weight,fiber_direction));

        FEM_Scaler E = YoungModulus;
        FEM_Scaler nu = PossonRatio;
        FEM_Scaler lambda = Enu2Lambda(E,nu);
        FEM_Scaler mu = Enu2Mu(E,nu);
        Mat9x9d dFactdF = EvaldFactdF(A_inv);

        phi = mu/2 * (Is[0] - I1_d) + lambda/2 * (Is[2] - 1) * (Is[2] - 1);
        dphi = mu/2 * (Ds[0] - I1_d_deriv) + lambda * (Is[2] - 1) * Ds[2];

        dphi = dFactdF.transpose() * dphi;

    }
    void ComputePhiDerivHessian(const Mat3x3d& Act,
            const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
            const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
            const Mat3x3d &F,FEM_Scaler& phi,Vec9d &dphi, Mat9x9d &ddphi,bool enforcing_spd = true) const override{
        Vec3d Is;
        std::vector<Vec9d> Ds(3);
        std::vector<Mat9x9d> Hs(3);

        Mat3x3d A_inv = Act.inverse();
        Mat3x3d F_act = F * A_inv;

        ComputeAnisotropicInvarientsDerivHessian(fiber_direction,aniso_weight,F_act,Is,Ds,Hs);

        // std::cout << "F_act : \n" << F_act << std::endl;
        // throw std::runtime_error("F_actcheck");

        FEM_Scaler I1_d = evalI1_delta(aniso_weight,fiber_direction,F_act);
        Vec9d I1_d_deriv = MatHelper::VEC(evalI1_delta_deriv(aniso_weight,fiber_direction));
        Mat9x9d dFactdF = EvaldFactdF(A_inv);

        FEM_Scaler E = YoungModulus;
        FEM_Scaler nu = PossonRatio;
        FEM_Scaler lambda = Enu2Lambda(E,nu);
        FEM_Scaler mu = Enu2Mu(E,nu);
        phi = mu/2 * (Is[0] - I1_d) + lambda/2 * (Is[2] - 1) * (Is[2] - 1);
        dphi = mu/2 * (Ds[0] - I1_d_deriv) + lambda * (Is[2] - 1) * Ds[2];
        dphi = dFactdF.transpose() * dphi;
        ddphi = mu/2 * Hs[0] + lambda * MatHelper::DYADIC(Ds[2],Ds[2]) + lambda * (Is[2] - 1) * Hs[2];
        ddphi = dFactdF.transpose() * ddphi * dFactdF;   
    }

public:
    /**
     * @brief The traditional Neohookean model is formulated using Cauchy-Green Invarients, but this set of invarients can not
     * cover all the isotropic material, take Corated-linear model for example. The Smith Invarients is a supper set of Cauchy-Green
     * invarients and can describe all the isotropic material.
     * @param _orient The orientation of the fibers
     * @param _weight the activation level of the three fiber direction
     * @param _S the stretching matrix
     * @param invarients the anisotropic invarients output
     * @see ComputeCGInvarients
     */
    inline void ComputeAnisotrpicInvarients(const Mat3x3d &_orient,
        const Vec3d& _weight,
        const Mat3x3d& _F,
        Vec3d& invarients) const{
            invarients.setZero();
            FEM_Scaler weight_sum = 0;
            for(size_t i = 0;i < 3;++i){
                invarients[0] += _weight[i] * _weight[i] * EvalI1(_F,_orient.col(i));
                weight_sum += _weight[i] * _weight[i];
            }
            invarients[0] /= weight_sum;
            invarients[0] *= 3;
            weight_sum = 0;
            for(size_t i = 0;i < 3;++i)
                for(size_t j = i+1;j < 3;++j){
                    weight_sum += _weight[i] * _weight[j];
                    invarients[1] = _weight[i] * _weight[j] * EvalI2(_F,_orient.col(i),_orient.col(j));
                }
            invarients[1] /= weight_sum;
            invarients[2] = EvalI3(_F);
    }  

    inline void ComputeAnisotropicInvarientsDeriv(const Mat3x3d &_orient,
        const Vec3d& _weight,
        const Mat3x3d& F,
        Vec3d& invarients,
        std::vector<Vec9d>& derivs) const{
            invarients.setZero();
            assert(derivs.size() == 3);
            std::fill(derivs.begin(),derivs.end(),Vec9d::Zero());
            FEM_Scaler weight_sum = 0;
            Vec9d buffer;
            for(size_t i = 0;i < 3;++i){
                FEM_Scaler weight = _weight[i] * _weight[i];
                invarients[0] += weight * EvalI1Deriv(F,_orient.col(i),buffer);
                derivs[0] += weight * buffer;
                // std::cout << "ORIENT : " << _orient.col(i).transpose() << std::endl;
                // std::cout << "ADD_BUFFER<" << i << "> : " << std::endl << MatHelper::MAT(buffer) << std::endl;
                weight_sum += weight;
            }
            invarients[0] /= weight_sum;
            invarients[0] *= 3;
            derivs[0] /= weight_sum;
            derivs[0] *= 3;

            weight_sum = 0;
            for(size_t i = 0;i < 3;++i)
                for(size_t j = i+1;j < 3;++j){
                    FEM_Scaler weight = _weight[i] * _weight[j];
                    weight_sum += weight;
                    invarients[1] += weight * EvalI2Deriv(F,_orient.col(i),_orient.col(j),buffer);
                    derivs[1] += weight * buffer;
                }
            invarients[1] /= weight_sum;
            derivs[1] /= weight_sum;

            invarients[2] = EvalI3Deriv(F,derivs[2]);      
    }

    inline void ComputeAnisotropicInvarientsDerivHessian(const Mat3x3d &_orient,
        const Vec3d& _weight,
        const Mat3x3d& F,
        Vec3d& invarients,
        std::vector<Vec9d>& derivs,
        std::vector<Mat9x9d>& Hs) const{
            invarients.setZero();
            assert(derivs.size() == 3);
            std::fill(derivs.begin(),derivs.end(),Vec9d::Zero());
            assert(Hs.size() == 3);
            std::fill(Hs.begin(),Hs.end(),Mat9x9d::Zero());

            FEM_Scaler weight_sum = 0;
            Vec9d D_buffer;
            Mat9x9d H_buffer;
            for(size_t i = 0;i < 3;++i){
                FEM_Scaler weight = _weight[i] * _weight[i];
                invarients[0] += weight * EvalI1DerivHessian(F,_orient.col(i),D_buffer,H_buffer);
                derivs[0] += weight* D_buffer;
                Hs[0] += weight * H_buffer;
                weight_sum += weight;
            }
            invarients[0] /= weight_sum;
            invarients[0] *= 3;
            derivs[0] /= weight_sum;
            derivs[0] *= 3;
            Hs[0] /= weight_sum;
            Hs[0] *= 3;

            weight_sum = 0;
            for(size_t i = 0;i < 3;++i)
                for(size_t j = i+1;j < 3;++j){
                    FEM_Scaler weight = _weight[i] * _weight[j];
                    weight_sum += weight;
                    invarients[1] += weight * EvalI2DerivHessian(F,_orient.col(i),_orient.col(j),D_buffer,H_buffer);
                    derivs[1] += weight * D_buffer;
                    Hs[1] += weight * H_buffer;
                }
            invarients[1] /= weight_sum;
            derivs[1] /= weight_sum;
            Hs[1] /= weight_sum;
            invarients[2] = EvalI3DerivHessian(F,derivs[2],Hs[2]);              
    }


    inline FEM_Scaler EvalI1(const Mat3x3d& F,const Vec3d& a) const {
        Vec3d a_ = F * a;
        return a_.squaredNorm();
    }

    inline FEM_Scaler EvalI1Deriv(const Mat3x3d& F,const Vec3d& a,Vec9d& deriv) const {
        FEM_Scaler I1 = EvalI1(F,a);
        deriv = 2 * MatHelper::VEC(F * MatHelper::DYADIC(a,a));

        return I1;
    }

    inline FEM_Scaler EvalI1DerivHessian(const Mat3x3d& F,const Vec3d& a,Vec9d& deriv,Mat9x9d& H) const {
        FEM_Scaler I1 = EvalI1Deriv(F,a,deriv);

        H.setZero();

        Mat3x3d dyadic_aa = MatHelper::DYADIC(a,a);
        for(size_t i = 0;i < 3;++i)
            for(size_t j = 0;j < 3;++j)
                H.block<3,3>(i*3,j*3).diagonal().setConstant(dyadic_aa(i,j));

        H *= 2;
        return I1;       
    }

    inline FEM_Scaler EvalI2(const Mat3x3d& F,const Vec3d& a1,const Vec3d& a2) const {
        Vec3d a1_ = F * a1;
        Vec3d a2_ = F * a2;
        FEM_Scaler I2 = a1_.dot(a2_);
        return I2;
    }

    inline FEM_Scaler EvalI2Deriv(const Mat3x3d& F,const Vec3d& a1,const Vec3d& a2,Vec9d& deriv) const {
        FEM_Scaler I2 = EvalI2(F,a1,a2);
        Mat3x3d dyadic_a12 = MatHelper::DYADIC(a1,a2);
        dyadic_a12 = (dyadic_a12 + dyadic_a12.transpose());
        deriv = MatHelper::VEC(F * dyadic_a12);

        return I2;
    }

    inline FEM_Scaler EvalI2DerivHessian(const Mat3x3d& F,const Vec3d& a1,const Vec3d& a2,Vec9d& deriv,Mat9x9d& H) const {
        FEM_Scaler I2 = EvalI2Deriv(F,a1,a2,deriv);
        H.setZero();

        Mat3x3d dyadic_a12 = MatHelper::DYADIC(a1,a2);
        dyadic_a12 = (dyadic_a12 + dyadic_a12.transpose());

        for(size_t i = 0;i < 3;++i)
            for(size_t j = 0;j < 3;++j)
                H.block<3,3>(i*3,j*3).diagonal().setConstant(dyadic_a12(i,j));     
        return I2;
    }    

    inline FEM_Scaler EvalI3(const Mat3x3d& F) const {
        return F.determinant();
    }

    inline FEM_Scaler EvalI3Deriv(const Mat3x3d& F,Vec9d& deriv) const {
        FEM_Scaler I3 = EvalI3(F);
        Vec3d f12 = F.col(1).cross(F.col(2));
        Vec3d f20 = F.col(2).cross(F.col(0));
        Vec3d f01 = F.col(0).cross(F.col(1));
        deriv << f12,f20,f01;

        return I3;
    }

    inline FEM_Scaler EvalI3DerivHessian(const Mat3x3d& F,Vec9d& deriv,Mat9x9d& H) const {
        FEM_Scaler I3 = EvalI3Deriv(F,deriv);
        Mat3x3d f0,f1,f2;
        f0 = MatHelper::ASYM(F.col(0));
        f1 = MatHelper::ASYM(F.col(1));
        f2 = MatHelper::ASYM(F.col(2));

        H.setZero();
        H.block<3,3>(3*1,3*0) = f2;
        H.block<3,3>(3*0,3*1) = -f2;
        H.block<3,3>(3*2,3*0) = -f1;
        H.block<3,3>(3*0,3*2) = f1;
        H.block<3,3>(3*2,3*1) = f0;
        H.block<3,3>(3*1,3*2) = -f0;

        return I3;
    }

    Mat3x3d get_I1_Sigma(const Vec3d& al) const {
        Mat3x3d Sigma;
        Sigma << al[0]*al[0],0,0,
                 0,al[1]*al[1],0,
                 0,0,al[2]*al[2];
        return Sigma;
    }

    Mat3x3d get_I2_Sigma(const Vec3d& al) const {
        Mat3x3d Sigma_hat;
        Sigma_hat << 0,al[0]*al[1],al[0]*al[2],
                     al[1]*al[0],0,al[1]*al[2],
                     al[2]*al[0],al[2]*al[1],0;
        return Sigma_hat;
    }

    FEM_Scaler evalI1_delta(const Vec3d& aw,const Mat3x3d& fiber_dir,const Mat3x3d& F) const {
        Mat3x3d dM = fiber_dir.transpose() * F * fiber_dir * get_I1_Sigma(aw);
        return 2*3*dM.trace()/aw.squaredNorm();
    }

    Mat3x3d evalI1_delta_deriv(const Vec3d& aw,const Mat3x3d& fiber_dir) const {
        Mat3x3d dM = fiber_dir * get_I1_Sigma(aw) * fiber_dir.transpose();
        return 2*3*dM/aw.squaredNorm();
    }
};
