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
     * @param forient the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     */
    void ComputePsi(const TetAttributes& attrs,const Mat3x3d& F,FEM_Scaler& psi) const override {
            FEM_Scaler iso_psi = 0;
            StableIsotropicMuscle::ComputePsi(attrs,F,iso_psi);

            FEM_Scaler Ia;
            Mat3x3d ActInv = attrs.emp.Act.inverse();
            Mat3x3d FAct = F * ActInv;

            Vec3d a = attrs.emp.forient;

            EvalAnisoInvarients(FAct,a,Ia);    
            FEM_Scaler Is = EvalReflection(FAct,a);
            FEM_Scaler mu = Enu2Mu(attrs.emp.E,attrs.emp.nu);

            FEM_Scaler aniso_psi = fiber_strength * mu / 2 * pow(sqrt(Ia) - Is,2);
            psi = iso_psi + aniso_psi;
    }
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination and element-wise energy gradient.
     * @param Act the activation level along the three orthogonal fiber directions
     * @param forient the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    void ComputePsiDeriv(const TetAttributes& attrs,const Mat3x3d& F,FEM_Scaler &psi,Vec9d &dpsi) const override {
            FEM_Scaler iso_psi,aniso_psi;
            Vec9d iso_dpsi,aniso_dpsi;

            StableIsotropicMuscle::ComputePsiDeriv(attrs,F,iso_psi,iso_dpsi);

            Mat3x3d ActInv = attrs.emp.Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler Ia;
            Vec9d ga;
            Vec3d a = attrs.emp.forient;

            EvalAnisoInvarientsDeriv(FAct,a,Ia,ga);   
            FEM_Scaler Is = EvalReflection(FAct,a);

            FEM_Scaler mu = Enu2Mu(attrs.emp.E,attrs.emp.nu);

            Mat9x9d dFactdF = EvaldFactdF(ActInv); 

            aniso_psi = fiber_strength * mu / 2 * pow(sqrt(Ia) - Is,2);
            aniso_dpsi = (fiber_strength * mu * (sqrt(Ia) - Is) / 2 / sqrt(Ia)) * ga;
            aniso_dpsi = dFactdF.transpose() * aniso_dpsi;


            psi = aniso_psi + iso_psi;
            dpsi = iso_dpsi + dFactdF.transpose() * aniso_dpsi;   
    }

    void ComputePsiDerivHessian(const TetAttributes& attrs,const Mat3x3d &F,FEM_Scaler& psi,Vec9d &dpsi, Mat9x9d &ddpsi,bool spd = true) const override{
            FEM_Scaler iso_psi,aniso_psi;
            Vec9d iso_dpsi,aniso_dpsi;
            Mat9x9d iso_ddpsi,aniso_ddpsi;

            Vec3d aniso_eigen_vals;
            Vec9d aniso_eigen_vecs[3];

            StableIsotropicMuscle::ComputePsiDerivHessian(attrs,F,iso_psi,iso_dpsi,iso_ddpsi,spd);

            Mat3x3d ActInv = attrs.emp.Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler Ia;
            Vec9d ga;

            Vec3d a = attrs.emp.forient;

            EvalAnisoInvarientsDeriv(FAct,a,Ia,ga);    
            FEM_Scaler Is = EvalReflection(FAct,a);

            FEM_Scaler mu = Enu2Mu(attrs.emp.E,attrs.emp.nu);

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
            
            Vec3d ahat = V.transpose() * a;

            Mat3x3d Q1,Q2;
            Q1 = U * Ts[0] * s.asDiagonal() * V.transpose() * A;
            Q2 = s[1] * ahat[1] * U * Ts[2] * s.asDiagonal() * V.transpose() * A - \
                    (s[2] * ahat[2]) * U * Ts[1] * s.asDiagonal() * V.transpose() * A;

            aniso_eigen_vecs[0] = MatHelper::VEC(Q0);
            aniso_eigen_vecs[1] = MatHelper::VEC(Q1);
            aniso_eigen_vecs[2] = MatHelper::VEC(Q2);

            for(size_t i = 0;i < 3;++i){
                if(aniso_eigen_vecs[i].norm() > 1e-6)
                    aniso_eigen_vecs[i] /= aniso_eigen_vecs[i].norm();
            }

            if(spd){
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

            if(std::isnan(psi)){
                std::cout << "NAN PSI DETECTED: : " << psi << "\t" << aniso_psi << "\t" << iso_psi << std::endl;
                throw std::runtime_error("NAN_PSI");
            }
            if(std::isnan(dpsi.norm())){
                std::cout << "NAN DPSI DETECTED : " << dpsi.norm() << "\t" << aniso_dpsi.norm() << "\t" << iso_dpsi.norm() << std::endl;
                throw std::runtime_error("NAN_DPSI");
            }
            if(std::isnan(ddpsi.norm())){
                std::cout << "NAN DDPSI DETECTED : " << dFactdF.norm() << "\t" << ddpsi.norm() << "\t" << aniso_ddpsi.norm() << "\t" << iso_ddpsi.norm() << std::endl;
                std::cout << "ANISOTROPIC EIGEN SPACE" << std::endl;
                for(size_t i = 0;i < 3;++i) {
                    std::cout << "EIGVAL : " << aniso_eigen_vals[i] << "\t" << aniso_eigen_vecs[i].norm() << std::endl;
                }

                std::cout << "U : " << std::endl << U << std::endl \
                    << "Ts[0]" << std::endl << Ts[0] << std::endl \
                    << "s:" << "\t" << s.transpose() << std::endl \
                    << "V : " << std::endl << V << std::endl \
                    << "A : " << std::endl << A << std::endl;

                std::cout << "Ts[1]" << std::endl << Ts[1] << std::endl;

                std::cout << "Q0 : " << std::endl << Q0 << std::endl;
                std::cout << "Q1 : " << std::endl << Q1 << std::endl;
                std::cout << "Q2 : " << std::endl << Q2 << std::endl;
                
                throw std::runtime_error("NAN_DDPSI");
            }
    }        

    void ComputePrincipalStress(const TetAttributes& attrs,const Vec3d& pstrain,Vec3d& pstress) const override {
        throw std::runtime_error("ANISOTROPIC_MODEL MIGHT BE PROBLEMATIC HERE");
    }

    void ComputePrincipalStressJacobi(const TetAttributes& attrs,const Vec3d& strain,Vec3d& stress,Mat3x3d& Jac) const override {
        throw std::runtime_error("ANISO_NH NOT IMPLEMENTED YET");
    }

private:
    FEM_Scaler fiber_strength;
    Mat3x3d Ts[3];
};