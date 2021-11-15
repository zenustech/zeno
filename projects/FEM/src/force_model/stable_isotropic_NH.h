#pragma once

#include "base_elastic_model.h"

/**
 * @class <StableIsotropicMuscle>
 * @brief A force model base class for inheritance, which defines the common interfaces needed for defining all the force model derived classes 
 * 
 * The current version only support isotropic elasto and damping model. The force model base class is unaware of the TetMesh, the input to all its
 * function is deformation gradient of individual element, instead of its deformaed shape.
 */
class StableIsotropicMuscle : public ElasticModel {
public:
    /**
     * @brief The constructor of StableIsotropicMuscle class
     */
    StableIsotropicMuscle() : ElasticModel() {}
    /**
     * @brief destructor method.
     */
    virtual ~StableIsotropicMuscle(){}


    /**
     * @brief An interface for defining the potential psi of anisotropic force model, all the force models should inherit this method and implement their 
     * own version of element-wise potential psi defination.
     * @param activation the activation level along the fiber direction
     * @param forient the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param psi the potential psi output
     */
    void ComputePhi(const ElastoMaterialParam& mp,const Mat3x3d& F,FEM_Scaler& psi) const override {
            // std::cout << "SNH EVAL PSI" << std::endl;
            Mat3x3d ActInv = mp.Act.inverse();
            Mat3x3d FAct = F * ActInv;
            FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
            FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);
            Vec3d Is;
            EvalIsoInvarients(FAct,Is);
            eval_phi(lambda,mu,Is,psi);
    }

    /**
     * @brief An interface for defining the potential psi of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential psi defination and element-wise psi gradient.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param forient the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param psi the potential psi output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    void ComputePhiDeriv(const ElastoMaterialParam& mp,const Mat3x3d& F,FEM_Scaler &psi,Vec9d &dpsi) const override {
            // std::cout << "SNH EVAL PSI DERIV" << std::endl;
            Mat3x3d ActInv = mp.Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
            FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);

            Vec3d Is;
            std::array<Vec9d,3> gs;
            EvalIsoInvarientsDeriv(FAct,Is,gs);

            eval_phi(lambda,mu,Is,psi);

            Vec3d dphi;
            eval_dphi(lambda,mu,Is,dphi);


            Mat9x9d dFactdF = EvaldFactdF(ActInv); 

            dpsi = dphi[0] * gs[0] + dphi[1] * gs[1] + dphi[2] * gs[2];     
            dpsi = dFactdF.transpose() * dpsi;
    }

    void ComputeIsoEigenSystem(FEM_Scaler lambda,FEM_Scaler mu,
        const Mat3x3d& F,
        Vec9d& eigen_vals,
        Vec9d eigen_vecs[9]) const {
        Mat3x3d U,V;
        Vec3d s;

        DiffSVD::SVD_Decomposition(F, U, s, V);

        Vec3d Is;
        EvalIsoInvarients(s, Is);
        Vec3d l_scale, l_twist, l_flip;

        Mat3x3d A;

        ComputeIsoStretchingMatrix(lambda, mu, Is, s, A);


        Mat3x3d U_proj;
        DiffSVD::SYM_Eigen_Decomposition(A, l_scale, U_proj);

        size_t offset = 0;
        //scale
        eigen_vals[offset++] = l_scale[0];
        eigen_vals[offset++] = l_scale[1];
        eigen_vals[offset++] = l_scale[2];
        // flip
        eigen_vals[offset++] = mu + s[2] * (lambda * (Is[2] - 1) - mu);
        eigen_vals[offset++] = mu + s[0] * (lambda * (Is[2] - 1) - mu);
        eigen_vals[offset++] = mu + s[1] * (lambda * (Is[2] - 1) - mu);
        //twist     
        eigen_vals[offset++] = mu - s[2] * (lambda * (Is[2] - 1) - mu);
        eigen_vals[offset++] = mu - s[0] * (lambda * (Is[2] - 1) - mu);
        eigen_vals[offset++] = mu - s[1] * (lambda * (Is[2] - 1) - mu);

        Mat9x3d proj_space;
        for (size_t i = 0; i < 3; ++i)
            proj_space.col(i) = MatHelper::VEC(U * Qs[i] * V.transpose());    

        for (size_t i = 0; i < 3; ++i)
            eigen_vecs[i] = proj_space * U_proj.col(i);

        // compute the flip and twist eigen matrix
        for (size_t i = 3; i < 9; ++i) {
            Mat3x3d Qi = U * Qs[i] * V.transpose();
            eigen_vecs[i] = MatHelper::VEC(Qi);
        }

    }
    /**
     * @brief An interface for defining the potential psi of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential psi defination, element-wise psi gradient and 12x12 element-wise psi hessian w.r.t deformed shape.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param forient the three orthogonal fiber directions
     * @param <F> the deformation gradient
     * @param <psi> the potential psi output
     * @param <derivative> the derivative of potential psi w.r.t the deformation gradient
     * @param <Hessian> the hessian of potential psi w.r.t the deformed shape for elasto model or nodal velocities for damping model
     * @param <spd> decide whether we should enforce the SPD of hessian matrix
     */
    void ComputePhiDerivHessian(const ElastoMaterialParam& mp,const Mat3x3d &F,FEM_Scaler& psi,Vec9d &dpsi, Mat9x9d &ddpsi,bool spd = true) const override {
        Mat3x3d ActInv = mp.Act.inverse();
        Mat3x3d FAct = F * ActInv;

        FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
        FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);

        Vec3d Is;
        std::array<Vec9d,3> gs;
        EvalIsoInvarientsDeriv(FAct,Is,gs);

        eval_phi(lambda,mu,Is,psi);

        Vec3d dphi;
        eval_dphi(lambda,mu,Is,dphi);

        Mat9x9d dFactdF = EvaldFactdF(ActInv); 

        dpsi = dphi[0] * gs[0] + dphi[1] * gs[1] + dphi[2] * gs[2];     
        dpsi = dFactdF.transpose() * dpsi;

        Vec9d eigen_vecs[9];
        Vec9d eigen_vals;
            
        // compute the eigen system of Hessian

        ComputeIsoEigenSystem(lambda,mu, FAct, eigen_vals, eigen_vecs);  

        if (spd) {
            for (size_t i = 0; i < 9; ++i)
                eigen_vals[i] = eigen_vals[i] > 1e-12 ? eigen_vals[i] : 1e-12;
        }
        ddpsi.setZero();
        for(size_t i = 0;i < 9;++i){
            ddpsi += eigen_vals[i] * eigen_vecs[i] * eigen_vecs[i].transpose();
            // std::cout << "UPDATE<" << i << "> : " << std::endl;
            // std::cout << eigen_vals[i] * eigen_vecs[i] * eigen_vecs[i].transpose() << std::endl;
        }

        // std::cout << "OUTPUT DDPSI : " << std::endl << Hessian << std::endl;

        ddpsi = dFactdF.transpose() * ddpsi * dFactdF; 
    }

    void ComputePrincipalStress(const ElastoMaterialParam& mp,const Vec3d& strain,Vec3d& stress) const override {
        FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
        FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);
        FEM_Scaler J = strain[0] * strain[1] * strain[2];
        stress[0] = mu * strain[0] - mu * strain[1] * strain[2] + lambda * (J - 1) * strain[1] * strain[2];
        stress[1] = mu * strain[1] - mu * strain[0] * strain[2] + lambda * (J - 1) * strain[0] * strain[2];
        stress[2] = mu * strain[2] - mu * strain[0] * strain[1] + lambda * (J - 1) * strain[0] * strain[1];
    }

    void ComputePrincipalStressJacobi(const ElastoMaterialParam& mp,const Vec3d& strain,Vec3d& stress,Mat3x3d& Jac) const override {
        FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
        FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);
        FEM_Scaler J = strain[0] * strain[1] * strain[2];
        stress[0] = mu * strain[0] - mu * strain[1] * strain[2] + lambda * (J - 1) * strain[1] * strain[2];
        stress[1] = mu * strain[1] - mu * strain[0] * strain[2] + lambda * (J - 1) * strain[0] * strain[2];
        stress[2] = mu * strain[2] - mu * strain[0] * strain[1] + lambda * (J - 1) * strain[0] * strain[1];

        FEM_Scaler J00 = mu + lambda * strain[1] * strain[1] * strain[2] * strain[2];
        FEM_Scaler J11 = mu + lambda * strain[0] * strain[0] * strain[2] * strain[2];
        FEM_Scaler J22 = mu + lambda * strain[0] * strain[0] * strain[1] * strain[1];

        FEM_Scaler J01 = -mu * strain[2] + lambda * (J - 1) * strain[2] + lambda * strain[0] * strain[1] * strain[2] * strain[2];
        FEM_Scaler J02 = -mu * strain[1] + lambda * (J - 1) * strain[1] + lambda * strain[0] * strain[1] * strain[1] * strain[2];
        FEM_Scaler J12 = -mu * strain[0] + lambda * (J - 1) * strain[0] + lambda * strain[0] * strain[0] * strain[1] * strain[2];

        Jac <<  J00,J01,J02,
                J01,J11,J12,
                J02,J12,J22;
    }


private :
    void eval_phi(FEM_Scaler lambda,FEM_Scaler mu,const Vec3d& Is,FEM_Scaler& phi) const {
        phi = mu / 2 * (Is[1] - 3) - mu * (Is[2] - 1) + lambda / 2 * (Is[2] - 1) * (Is[2] - 1);
    }

    void eval_dphi(FEM_Scaler lambda,FEM_Scaler mu,const Vec3d& Is,Vec3d& dphi) const {
        dphi[0] = 0;
        dphi[1] = mu / 2;
        dphi[2] = -mu + lambda * (Is[2] - 1);
    }

    void eval_ddphi(FEM_Scaler lambda,FEM_Scaler mu,const Vec3d& Is,Mat3x3d& ddphi) const {
        ddphi <<    0,0,0,
                    0,0,0,
                    0,0,lambda;
    }

    void ComputeIsoStretchingMatrix(FEM_Scaler lambda,FEM_Scaler mu,const Vec3d& Is,const Vec3d& sigma,Mat3x3d& A) const {
        A(0,0) = mu + lambda * Is[2] * Is[2] / sigma[0] / sigma[0];
        A(0,1) = sigma[2] * (lambda * (2*Is[2] - 1) - mu);
        A(0,2) = sigma[1] * (lambda * (2*Is[2] - 1) - mu);
        A(1,1) = mu + lambda * Is[2] * Is[2] / sigma[1] / sigma[1];
        A(1,2) = sigma[0] * (lambda * (2*Is[2] - 1) - mu);
        A(2,2) = mu + lambda * Is[2] * Is[2] / sigma[2] / sigma[2];

        A(1,0) = A(0,1);
        A(2,0) = A(0,2);
        A(2,1) = A(1,2);
    }
};
