#pragma

#include "stable_isotropic_NH.h"


class StableAnisotropicMuscle : public StableIsotropicMuscle {
public:
    StableAnisotropicMuscle(FEM_Scaler aniso_strength) : StableIsotropicMuscle() {
        fiber_strength = aniso_strength;

        Ts[0] << 0, 0, 0,
                 0, 0, 1,
                 0,-1, 0;
        Ts[1] << 0, 0,-1,
                 0, 0, 0,
                 1, 0, 0;
        Ts[2] << 0, 1, 0,
                -1, 0, 0,
                 0, 0, 0;
    }
    /**
     * @brief destructor method.
     */
    virtual ~StableAnisotropicMuscle(){}
    /**
     * @brief An interface for defining the potential energy of anisotropic force model, all the force models should inherit this method and implement their 
     * own version of element-wise potential energy defination.
     * @param Act the activation level along the fiber direction
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     */
    void ComputePhi(const Mat3x3d& Act,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d& F,FEM_Scaler& energy) const override {
            FEM_Scaler iso_psi;
            StableIsotropicMuscle::ComputePhi(Act,aniso_weight,fiber_direction,YoungModulus,PossonRatio,F,iso_psi);

            FEM_Scaler Ia;
            Mat3x3d ActInv = Act.inverse();
            Mat3x3d FAct = F * ActInv;

            EvalAnisoInvarients(FAct,fiber_direction.col(0),Ia);    

            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler aniso_psi = fiber_strength * mu / 2 * pow(Ia - 1,2);

            energy = iso_psi + aniso_psi;
    }
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination and element-wise energy gradient.
     * @param Act the activation level along the three orthogonal fiber directions
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    void ComputePhiDeriv(const Mat3x3d& Act,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d& F,FEM_Scaler &energy,Vec9d &derivative) const override {
            FEM_Scaler iso_psi;
            Vec9d iso_gradient,aniso_gradient;

            StableIsotropicMuscle::ComputePhiDeriv(Act,aniso_weight,fiber_direction,YoungModulus,PossonRatio,F,iso_psi,iso_gradient);

            Mat3x3d ActInv = Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler Ia;
            Vec9d ga;

            EvalAnisoInvarientsDeriv(FAct,fiber_direction.col(0),Ia,ga);    

            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            FEM_Scaler aniso_psi = fiber_strength * mu / 2 * pow(Ia - 1,2);
            aniso_gradient = fiber_strength * mu * (Ia - 1) * ga;

            energy = aniso_psi + iso_psi;

            Mat9x9d dFactdF = EvaldFactdF(ActInv); 
            derivative = iso_gradient + dFactdF.transpose() * aniso_gradient;   
    }

    void ComputePhiDerivHessian(const Mat3x3d& Act,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
        const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d &F,FEM_Scaler& psi,Vec9d &dpsi, Mat9x9d &ddpsi,bool enforcing_spd = true) const override{
            FEM_Scaler iso_psi;
            Vec9d iso_dpsi,aniso_dpsi;

            StableIsotropicMuscle::ComputePhiDeriv(Act,aniso_weight,fiber_direction,YoungModulus,PossonRatio,F,iso_psi,iso_dpsi);

            Vec3d a = fiber_direction.col(0);

            Mat3x3d ActInv = Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler Ia;
            Vec9d ga;

            EvalAnisoInvarientsDeriv(FAct,fiber_direction.col(0),Ia,ga);    

            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            FEM_Scaler aniso_psi = fiber_strength * mu / 2 * pow(Ia - 1,2);
            aniso_dpsi = fiber_strength * mu * (Ia - 1) * ga;

            psi = aniso_psi + iso_psi;

            Mat9x9d dFactdF = EvaldFactdF(ActInv); 
            dpsi = iso_dpsi + dFactdF.transpose() * aniso_dpsi;   


            Vec9d iso_eigen_vecs[9];
            Vec9d iso_eigen_vals;                
            ComputeIsoEigenSystem(YoungModulus,PossonRatio, FAct, iso_eigen_vals, iso_eigen_vecs);  


        //     enforcing_spd = false;

            if (enforcing_spd) {
                for (size_t i = 0; i < 9; ++i)
                        iso_eigen_vals[i] = iso_eigen_vals[i] > 0 ? iso_eigen_vals[i] : 0;
                }
            
            FEM_Scaler smallest_iso_val = iso_eigen_vals.minCoeff();

            // compute anisotropic eigen system
            Vec3d aniso_eigen_vals;
            Vec9d aniso_eigen_vecs[3];

            aniso_eigen_vals[0] = 2 * fiber_strength * mu * ( 3 * Ia - 1 );
            aniso_eigen_vals[1] = aniso_eigen_vals[2] = 2 * fiber_strength * (Ia - 1);

            Mat3x3d A = MatHelper::DYADIC(a,a);
            Mat3x3d Q0 = 1 / sqrt(Ia) * FAct * MatHelper::DYADIC(a,a);
            Mat3x3d U,V;
            Vec3d s;
            DiffSVD::SVD_Decomposition(FAct,U,s,V);
            
            Mat3x3d Q1 = U * Ts[0] * s.asDiagonal() * V.transpose() * A;
            Vec3d ahat = V.transpose() * a;

            Mat3x3d Q2 = s[1] * ahat[1] * U * Ts[2] * s.asDiagonal() * V.transpose() * A - \
                (s[2] * ahat[2]) * U * Ts[1] * s.asDiagonal() * V.transpose() * A;

            aniso_eigen_vecs[0] = MatHelper::VEC(Q0);
            aniso_eigen_vecs[1] = MatHelper::VEC(Q1);
            aniso_eigen_vecs[2] = MatHelper::VEC(Q2);

            if(enforcing_spd){
                for(size_t i = 0;i < 3;++i)
                    aniso_eigen_vals[i] = aniso_eigen_vals[i] < 0 ? 0 : aniso_eigen_vals[i];
            }

            ddpsi.setZero();
            for(size_t i = 0;i < 9;++i){
                ddpsi += iso_eigen_vals[i] * iso_eigen_vecs[i] * iso_eigen_vecs[i].transpose();
            }

            for(size_t i = 0;i < 3;++i)
                ddpsi += aniso_eigen_vals[i] * aniso_eigen_vecs[i] * aniso_eigen_vecs[i].transpose();

            ddpsi = dFactdF.transpose() * ddpsi *dFactdF;

        //     Mat9x9d ddpsi_fd = Mat9x9d::Zero();
        //     FEM_Scaler step = 1e-5;
        //     for(size_t i = 0;i < 9;++i){
        //         Mat3x3d F_tmp = F;
        //         Vec9d f_tmp = MatHelper::VEC(F_tmp);
        //         f_tmp[i] += step;
        //         F_tmp = MatHelper::MAT(f_tmp);

        //         FEM_Scaler psi_tmp;
        //         Vec9d dpsi_tmp;
        //         ComputePhiDeriv(Act,aniso_weight,fiber_direction,YoungModulus,PossonRatio,F_tmp,psi_tmp,dpsi_tmp);

        //         ddpsi_fd.col(i) = (dpsi_tmp - dpsi) / step;
        //     }

        //     FEM_Scaler ddpsi_error = (ddpsi_fd - ddpsi).norm() / ddpsi_fd.norm();
        // //     if(ddpsi_error > 1e-3){
        //         std::cerr << "ddpsi_error : " << ddpsi_error << std::endl;
        //         std::cout << "ddpsi : " << std::endl << ddpsi << std::endl;
        //         std::cout << "ddpsi_fd : " << std::endl << ddpsi_fd << std::endl;
        //         throw std::runtime_error("ddpsi_error");
        // //     }

        //     throw std::runtime_error("ddpsi check");
    }        

private:
    FEM_Scaler fiber_strength;
    Mat3x3d Ts[3];
};