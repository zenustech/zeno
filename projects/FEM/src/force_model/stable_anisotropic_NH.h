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

    FEM_Scaler EvalReflection(const Mat3x3d& F,const Vec3d& a) const {
        Mat3x3d R,S;
        DiffSVD::Polar_Decomposition(F,R,S);
        FEM_Scaler sign = a.transpose() * S * a;
        if(fabs(sign) < 1e-5)
                return 0;
        return sign > 0 ? 1 : -1;
    }

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
        const Mat3x3d& F,FEM_Scaler& psi) const override {
            FEM_Scaler iso_psi = 0;
            StableIsotropicMuscle::ComputePhi(Act,aniso_weight,fiber_direction,YoungModulus,PossonRatio,F,iso_psi);

            FEM_Scaler Ia;
            Mat3x3d ActInv = Act.inverse();
            Mat3x3d FAct = F * ActInv;

            Vec3d a = fiber_direction.col(0);

            EvalAnisoInvarients(FAct,a,Ia);    
            FEM_Scaler Is = EvalReflection(FAct,a);
            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);

            FEM_Scaler aniso_psi = fiber_strength * mu / 2 * pow(sqrt(Ia) - Is,2);
            psi = iso_psi + aniso_psi;
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
        const Mat3x3d& F,FEM_Scaler &psi,Vec9d &dpsi) const override {
            FEM_Scaler iso_psi,aniso_psi;
            Vec9d iso_dpsi,aniso_dpsi;

            StableIsotropicMuscle::ComputePhiDeriv(Act,aniso_weight,fiber_direction,YoungModulus,PossonRatio,F,iso_psi,iso_dpsi);

            Mat3x3d ActInv = Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler Ia;
            Vec9d ga;
            Vec3d a = fiber_direction.col(0);

            EvalAnisoInvarientsDeriv(FAct,a,Ia,ga);   
            FEM_Scaler Is = EvalReflection(FAct,a);

            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            Mat9x9d dFactdF = EvaldFactdF(ActInv); 

            aniso_psi = fiber_strength * mu / 2 * pow(sqrt(Ia) - Is,2);
            aniso_dpsi = (fiber_strength * mu * (sqrt(Ia) - Is) / 2 / sqrt(Ia)) * ga;
            aniso_dpsi = dFactdF.transpose() * aniso_dpsi;


            psi = aniso_psi + iso_psi;
            dpsi = iso_dpsi + dFactdF.transpose() * aniso_dpsi;   
    }

    void ComputePhiDerivHessian(const Mat3x3d& Act,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,
        const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d &F,FEM_Scaler& psi,Vec9d &dpsi, Mat9x9d &ddpsi,bool enforcing_spd) const override{
            FEM_Scaler iso_psi,aniso_psi;
            Vec9d iso_dpsi,aniso_dpsi;
            Mat9x9d iso_ddpsi,aniso_ddpsi;

            Vec3d aniso_eigen_vals;
            Vec9d aniso_eigen_vecs[3];

            StableIsotropicMuscle::ComputePhiDerivHessian(Act,aniso_weight,
                fiber_direction,YoungModulus,PossonRatio,
                F,iso_psi,iso_dpsi,iso_ddpsi,enforcing_spd);

            Mat3x3d ActInv = Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler Ia;
            Vec9d ga;

            Vec3d a = fiber_direction.col(0);

            EvalAnisoInvarientsDeriv(FAct,a,Ia,ga);    
            FEM_Scaler Is = EvalReflection(FAct,a);

            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            Mat9x9d dFactdF = EvaldFactdF(ActInv);

            aniso_psi = fiber_strength * mu / 2 * pow(sqrt(Ia) - Is,2);
            aniso_dpsi = fiber_strength * mu * (sqrt(Ia) - Is) / 2 / sqrt(Ia) * ga;
            aniso_dpsi = dFactdF.transpose() * aniso_dpsi;

            aniso_eigen_vals[0] = fiber_strength * mu;
            aniso_eigen_vals[1] = aniso_eigen_vals[2] = fiber_strength * mu * (1 - Is / sqrt(Ia));

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

            for(size_t i = 0;i < 3;++i)
                aniso_eigen_vecs[i] /= aniso_eigen_vecs[i].norm();

            if(enforcing_spd){
                for(size_t i = 0;i < 3;++i)
                    aniso_eigen_vals[i] = aniso_eigen_vals[i] < 1e-12 ? 1e-12 : aniso_eigen_vals[i];
            }

            aniso_ddpsi.setZero();
            for(size_t i = 0;i < 3;++i)
                aniso_ddpsi += aniso_eigen_vals[i] * aniso_eigen_vecs[i] * aniso_eigen_vecs[i].transpose();
            aniso_ddpsi = dFactdF.transpose() * aniso_ddpsi *dFactdF;

            psi = aniso_psi + iso_psi;
            dpsi = iso_dpsi + aniso_dpsi;   
            ddpsi = aniso_ddpsi + iso_ddpsi;
    }        

private:
    FEM_Scaler fiber_strength;
    Mat3x3d Ts[3];
};