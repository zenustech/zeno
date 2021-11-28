#pragma once

#include "base_elastic_model.h"

/**
 * @class <StableIsotropicMuscle>
 * @brief A force model base class for inheritance, which defines the common interfaces needed for defining all the force model derived classes 
 * 
 * The current version only support isotropic elasto and damping model. The force model base class is unaware of the TetMesh, the input to all its
 * function is deformation gradient of individual element, instead of its deformaed shape.
 */
class StableStvk : public ElasticModel {
public:
    /**
     * @brief The constructor of StableIsotropicMuscle class
     */
    StableStvk() : ElasticModel() {}
    /**
     * @brief destructor method.
     */
    virtual ~StableStvk(){}

    /**
     * @brief An interface for defining the potential energy of anisotropic force model, all the force models should inherit this method and implement their 
     * own version of element-wise potential energy defination.
     * @param activation the activation level along the fiber direction
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     */
    void ComputePhi(const ElastoMaterialParam& mp,const Mat3x3d& F,FEM_Scaler& psi) const override {
        Mat3x3d ActInv = mp.Act.inverse();
        Mat3x3d FAct = F * ActInv;

        FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
        FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);

        Mat3x3d CStrain = 0.5 * (FAct.transpose() * FAct - Mat3x3d::Identity());

        FEM_Scaler TE = CStrain.trace();
        psi = mu * CStrain.squaredNorm() + lambda / 2 * TE * TE;
    }
    /**
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination and element-wise energy gradient.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     * @param the derivative of potential w.r.t the deformed shape for elasto model or nodal velocities for damping model
     */
    void ComputePhiDeriv(const ElastoMaterialParam& mp,const Mat3x3d& F,FEM_Scaler &psi,Vec9d &dpsi) const override {
            Mat3x3d ActInv = mp.Act.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
            FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);

            Mat3x3d CStrain = 0.5 * (FAct.transpose() * FAct - Mat3x3d::Identity());

            FEM_Scaler TE = CStrain.trace();
            psi = mu * CStrain.squaredNorm() + lambda / 2 * TE * TE;

            Mat3x3d P = FAct * (2 * mu * CStrain + lambda * TE * Mat3x3d::Identity());

            Mat9x9d dFactdF = EvaldFactdF(ActInv); 
            dpsi = dFactdF.transpose() * MatHelper::VEC(P);
    }

    void ComputeIsoEigenSystem(FEM_Scaler lambda,FEM_Scaler mu,const Mat3x3d& F,Vec9d& eigen_vals,Vec9d eigen_vecs[9]) const {
        Mat3x3d U,V;
        Vec3d s;

        DiffSVD::SVD_Decomposition(F, U, s, V);

        Vec3d Is;
        EvalIsoInvarients(s, Is);

        // compute the eigen system of Hessian
        Vec3d l_scale, l_twist, l_flip;

        Mat3x3d A;

        ComputeIsoStretchingMatrix(lambda, mu, Is, s, A);

        // std::cout << "Stvk Stretching Matrix : " << std::endl << A << std::endl;

        Mat3x3d U_proj;
        DiffSVD::SYM_Eigen_Decomposition(A, l_scale, U_proj);

        size_t offset = 0;
        //scale
        eigen_vals[offset++] = l_scale[0];
        eigen_vals[offset++] = l_scale[1];
        eigen_vals[offset++] = l_scale[2];
        // flip
        eigen_vals[offset++] = (Is[1] - 3) * lambda / 2  + mu * (s[0]*s[0] + s[1]*s[1] - s[0]*s[1] - 1);
        eigen_vals[offset++] = (Is[1] - 3) * lambda / 2  + mu * (s[1]*s[1] + s[2]*s[2] - s[1]*s[2] - 1);
        eigen_vals[offset++] = (Is[1] - 3) * lambda / 2  + mu * (s[0]*s[0] + s[2]*s[2] - s[0]*s[2] - 1);
        //twist  
        eigen_vals[offset++] = (Is[1] - 3) * lambda / 2  + mu * (s[0]*s[0] + s[1]*s[1] + s[0]*s[1] - 1);   
        eigen_vals[offset++] = (Is[1] - 3) * lambda / 2  + mu * (s[1]*s[1] + s[2]*s[2] + s[1]*s[2] - 1);
        eigen_vals[offset++] = (Is[1] - 3) * lambda / 2  + mu * (s[0]*s[0] + s[2]*s[2] + s[0]*s[2] - 1);

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
     * @brief An interface for defining the potential energy of force model, all the force models should inherit this method and implement their
     * own version of element-wise potential energy defination, element-wise energy gradient and 12x12 element-wise energy hessian w.r.t deformed shape.
     * @param activation the activation level along the three orthogonal fiber directions
     * @param fiber_direction the three orthogonal fiber directions
     * @param <F> the deformation gradient
     * @param <energy> the potential energy output
     * @param <derivative> the derivative of potential energy w.r.t the deformation gradient
     * @param <Hessian> the hessian of potential energy w.r.t the deformed shape for elasto model or nodal velocities for damping model
     * @param <spd> decide whether we should enforce the SPD of hessian matrix
     */
    void ComputePhiDerivHessian(const ElastoMaterialParam& mp,const Mat3x3d &F,FEM_Scaler& psi,Vec9d &dpsi, Mat9x9d &ddpsi,bool spd = true) const override {
        Mat3x3d ActInv = mp.Act.inverse();
        Mat3x3d FAct = F * ActInv;

        FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
        FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);

        Mat3x3d CStrain = 0.5 * (FAct.transpose() * FAct - Mat3x3d::Identity());

        FEM_Scaler TE = CStrain.trace();
        psi = mu * CStrain.squaredNorm() + lambda / 2 * TE * TE;

        Mat3x3d P = FAct * (2 * mu * CStrain + lambda * TE * Mat3x3d::Identity());

        // bool check_principal_stress = true;
        // if(check_principal_stress){
        //     Mat3x3d Ftest = Mat3x3d::Random();
        //     Vec3d pstrain,pstress;
        //     Mat3x3d U,V;
        //     DiffSVD::SVD_Decomposition(Ftest,U,pstrain,V);
        //     ComputePrincipalStress(mp,pstrain,pstress);

        //     Mat3x3d CStrainTest = 0.5 * (Ftest.transpose() * Ftest - Mat3x3d::Identity());

        //     FEM_Scaler TETest = CStrainTest.trace();
        //     Mat3x3d PTest = Ftest * (2 * mu * CStrainTest + lambda * TETest * Mat3x3d::Identity());

        //     Mat3x3d Prc = U * pstress.asDiagonal() * V.transpose();

        //     std::cout << "PTest : " << std::endl << U.transpose() * PTest * V << std::endl << "Prc : " << std::endl << pstress << std::endl;

        //     throw std::runtime_error("PTest vs Prc");
        // }


        Mat9x9d dFactdF = EvaldFactdF(ActInv); 
        dpsi = dFactdF.transpose() * MatHelper::VEC(P);

        Vec3d Is;
        std::array<Vec9d,3> gs;

        EvalIsoInvarientsDeriv(FAct,Is,gs);

        Vec9d eigen_vecs[9];
        Vec9d eigen_vals;

        ComputeIsoEigenSystem(lambda,mu, FAct, eigen_vals, eigen_vecs); 

        if (spd) {
            for (size_t i = 0; i < 9; ++i)
                eigen_vals[i] = eigen_vals[i] > 1e-12 ? eigen_vals[i] : 1e-12;
        }
        ddpsi.setZero();
        for(size_t i = 0;i < 9;++i){
            ddpsi += eigen_vals[i] * eigen_vecs[i] * eigen_vecs[i].transpose();
        }

        ddpsi = dFactdF.transpose() * ddpsi * dFactdF; 
    }

    void ComputePrincipalStress(const ElastoMaterialParam& mp,const Vec3d& strain,Vec3d& stress) const override {
        FEM_Scaler mu = Enu2Mu(mp.E,mp.nu);
        FEM_Scaler lambda = Enu2Lambda(mp.E,mp.nu);

        const Vec3d &s = strain;
        stress[0] = mu * (s[0] * s[0] - 1) * s[0] + lambda * (s.squaredNorm() - 3) * s[0] / 2;
        stress[1] = mu * (s[1] * s[1] - 1) * s[1] + lambda * (s.squaredNorm() - 3) * s[1] / 2;
        stress[2] = mu * (s[2] * s[2] - 1) * s[2] + lambda * (s.squaredNorm() - 3) * s[2] / 2;
    }

    void ComputePrincipalStressJacobi(const ElastoMaterialParam& mp,const Vec3d& strain,Vec3d& stress,Mat3x3d& Jac) const override {
        FEM_Scaler lambda = ElasticModel::Enu2Lambda(mp.E,mp.nu);
        FEM_Scaler mu = ElasticModel::Enu2Mu(mp.E,mp.nu);

        const Vec3d &s = strain;
        stress[0] = mu * (s[0] * s[0] - 1) * s[0] + lambda * (s.squaredNorm() - 3) * s[0] / 2;
        stress[1] = mu * (s[1] * s[1] - 1) * s[1] + lambda * (s.squaredNorm() - 3) * s[1] / 2;
        stress[2] = mu * (s[2] * s[2] - 1) * s[2] + lambda * (s.squaredNorm() - 3) * s[2] / 2;

        Vec3d Is;
        EvalIsoInvarients(strain, Is);
        ComputeIsoStretchingMatrix(lambda,mu,Is,strain,Jac);
    }

private :
    void ComputeIsoStretchingMatrix(FEM_Scaler lambda,FEM_Scaler mu,const Vec3d& Is,const Vec3d& sigma,Mat3x3d& A) const {
        for(size_t i = 0;i < 3;++i)
            A(i,i) = -mu + lambda / 2 * (Is[1] - 3) + (lambda + 3 * mu) * sigma[i] * sigma[i];

        for(size_t i = 0;i < 3;++i)
            for(size_t j = i + 1;j < 3;++j)
                A(i,j) = A(j,i) = lambda * sigma[i] * sigma[j];
    }
};
