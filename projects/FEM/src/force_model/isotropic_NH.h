#pragma once

#include "base_force_model.h"

/**
 * @class <SmithSNHModel>
 * @brief A force model base class for inheritance, which defines the common interfaces needed for defining all the force model derived classes 
 * 
 * The current version only support isotropic elasto and damping model. The force model base class is unaware of the TetMesh, the input to all its
 * function is deformation gradient of individual element, instead of its deformaed shape.
 */
class SmithSNHModel : public BaseForceModel{
public:
    /**
     * @brief The constructor of SmithSNHModel class
     */
    SmithSNHModel() : BaseForceModel() {
        Qs[0] <<   1, 0, 0,
                0, 0, 0,
                0, 0, 0;
        Qs[1]  <<   0, 0, 0,
                0, 1, 0,
                0, 0, 0;
        Qs[2]  <<   0, 0, 0,
                0, 0, 0,
                0, 0, 1;

        Qs[3]  <<   0,-1, 0,
                1, 0, 0,
                0, 0, 0;
        Qs[3]  /= sqrt(2);
    
        Qs[4]  <<   0, 0, 0,
                0, 0, 1,
                0,-1, 0;
        Qs[4]  /= sqrt(2);

        Qs[5]  <<   0, 0, 1,
                0, 0, 0,
                -1,0, 0;
        Qs[5]  /= sqrt(2);

        Qs[6]  <<   0, 1, 0,
                1, 0, 0,
                0, 0, 0;
        Qs[6]  /= sqrt(2);

        Qs[7]  <<   0, 0, 0,
                0, 0, 1,
                0, 1, 0;
        Qs[7]  /= sqrt(2);

        Qs[8]  <<   0, 0, 1,
                0, 0, 0,
                1, 0, 0;
        Qs[8]  /= sqrt(2);
    }
    /**
     * @brief destructor method.
     */
    virtual ~SmithSNHModel(){}

    inline void ComputeSmithInvarients(const Vec3d& sigma,FEM_Scaler &I1,FEM_Scaler& I2,FEM_Scaler& I3) const{
        I1 = sigma.sum();
        I2 = sigma.squaredNorm();
        I3 = sigma[0] * sigma[1] * sigma[2];
    }

    /**
     * @brief An interface for defining the potential energy of anisotropic force model, all the force models should inherit this method and implement their 
     * own version of element-wise potential energy defination.
     * @param activation the activation level along the fiber direction
     * @param fiber_direction the three orthogonal fiber directions
     * @param F the deformation gradient
     * @param energy the potential energy output
     */
    void ComputePhi(const Mat3x3d& activation,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d& F,FEM_Scaler& energy) const override {
            Mat3x3d ActInv = activation.inverse();
            Mat3x3d FAct = F * ActInv;


            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            Mat3x3d U,V;
            Vec3d s;
            DiffSVD::SVD_Decomposition(FAct,U,s,V);

            Vec3d Is;
            ComputeSmithInvarients(s,Is[0],Is[1],Is[2]);
    
            eval_phi(lambda,mu,Is,energy);
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
    void ComputePhiDeriv(const Mat3x3d& activation,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d& F,FEM_Scaler &energy,Vec9d &derivative) const override {
            Mat3x3d ActInv = activation.inverse();
            Mat3x3d FAct = F * ActInv;

            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            Mat3x3d U,V;
            Vec3d s;
            DiffSVD::SVD_Decomposition(FAct, U, s, V);

            Vec3d Is;
            ComputeSmithInvarients(s,Is[0],Is[1],Is[2]);

            eval_phi(lambda,mu,Is,energy);

            Vec9d g1,g2,g3;
            Mat3x3d J;

            Mat3x3d R = U * V.transpose();

            g1 = MatHelper::VEC(R);
            g2 = 2 * MatHelper::VEC(FAct);

            J.col(0) = FAct.col(1).cross(FAct.col(2));
            J.col(1) = FAct.col(2).cross(FAct.col(0));
            J.col(2) = FAct.col(0).cross(FAct.col(1));

            g3 = MatHelper::VEC(J);

            Vec3d dphi;
            eval_dphi(lambda,mu,Is,dphi);


            Mat9x9d dFactdF = EvaldFactdF(ActInv); 

            derivative = dphi[0] * g1 + dphi[1] * g2 + dphi[2] * g3;     
            derivative = dFactdF.transpose() * derivative;
        }

        void ComputePhiHessianEigenSystem(FEM_Scaler YoungModulus,FEM_Scaler PossonRatio,
            const Mat3x3d& F,
            Vec9d& eigen_vals,
            Vec9d eigen_vecs[9]) const {
            FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
            FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

            Mat3x3d U,V;
            Vec3d s;
            DiffSVD::SVD_Decomposition(F, U, s, V);

            Vec3d Is;
            ComputeSmithInvarients(s, Is[0], Is[1], Is[2]);

            // compute the eigen system of Hessian
            Vec3d l_scale, l_twist, l_flip;

            Mat3x3d A;
            ComputeStretchingMatrix(lambda, mu, Is, s, A);

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
                // std::cout << i << " ortho w.r.t g3 :" << eigen_vecs[i].dot(g3) << std::endl;
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
     * @param <enforcing_spd> decide whether we should enforce the SPD of hessian matrix
     */
    void ComputePhiDerivHessian(const Mat3x3d& activation,
        const Vec3d& aniso_weight,const Mat3x3d& fiber_direction,const FEM_Scaler& YoungModulus,const FEM_Scaler& PossonRatio,
        const Mat3x3d &F,FEM_Scaler& energy,Vec9d &derivative, Mat9x9d &Hessian,bool enforcing_spd = true) const override {

        Mat3x3d ActInv = activation.inverse();
        Mat3x3d FAct = F * ActInv;
            
        Vec9d eigen_vecs[9];
        Vec9d eigen_vals;

        FEM_Scaler mu = Enu2Mu(YoungModulus,PossonRatio);
        FEM_Scaler lambda = Enu2Lambda(YoungModulus,PossonRatio);

        Mat3x3d U,V;
        Vec3d s;
        DiffSVD::SVD_Decomposition(FAct, U, s, V);

        Vec3d Is;
        ComputeSmithInvarients(s,Is[0],Is[1],Is[2]);

        eval_phi(lambda,mu,Is,energy);

        Vec9d g1,g2,g3;
        Mat3x3d J;

       Mat3x3d R = U * V.transpose();

        g1 = MatHelper::VEC(R);
        g2 = 2 * MatHelper::VEC(FAct);

        J.col(0) = FAct.col(1).cross(FAct.col(2));
        J.col(1) = FAct.col(2).cross(FAct.col(0));
        J.col(2) = FAct.col(0).cross(FAct.col(1));

        g3 = MatHelper::VEC(J);


        Mat9x9d dFactdF = EvaldFactdF(ActInv);   

        Vec3d dphi;
        eval_dphi(lambda,mu,Is,dphi);

        derivative = dphi[0] * g1 + dphi[1] * g2 + dphi[2] * g3; 
        derivative = dFactdF.transpose() * derivative;

        // compute the eigen system of Hessian
        ComputePhiHessianEigenSystem(YoungModulus,PossonRatio, FAct, eigen_vals, eigen_vecs);  
        if (enforcing_spd) {
            for (size_t i = 0; i < 9; ++i)
                eigen_vals[i] = eigen_vals[i] > 0 ? eigen_vals[i] : 0;
        }
        Hessian.setZero();
        for(size_t i = 0;i < 9;++i){
            Hessian += eigen_vals[i] * eigen_vecs[i] * eigen_vecs[i].transpose();
        }

        Hessian = dFactdF.transpose() * Hessian * dFactdF; 
    }

    static FEM_Scaler Enu2Lambda(FEM_Scaler E,FEM_Scaler nu) {return (nu * E)/((1 + nu) * (1 - 2*nu));}
    static FEM_Scaler Enu2Mu(FEM_Scaler E,FEM_Scaler nu) {return E / (2 * (1 + nu));}
    static FEM_Scaler Lame2E(FEM_Scaler lambda,FEM_Scaler mu) {return mu*(2*mu+3*lambda)/(mu+lambda);}
    static FEM_Scaler Lame2Nu(FEM_Scaler lambda,FEM_Scaler mu) {return lambda / (2 * (lambda +mu));}


    inline Mat9x9d EvaldFactdF(const Mat3x3d& Act_inv) const {
        Mat9x9d M = Mat9x9d::Zero();
        
        M(0,0) = M(1,1) = M(2,2) = Act_inv(0,0);
        M(3,0) = M(4,1) = M(5,2) = Act_inv(0,1);
        M(6,0) = M(7,1) = M(8,2) = Act_inv(0,2);

        M(0,3) = M(1,4) = M(2,5) = Act_inv(1,0);
        M(3,3) = M(4,4) = M(5,5) = Act_inv(1,1);
        M(6,3) = M(7,4) = M(8,5) = Act_inv(1,2);

        M(0,6) = M(1,7) = M(2,8) = Act_inv(2,0);
        M(3,6) = M(4,7) = M(5,8) = Act_inv(2,1);
        M(6,6) = M(7,7) = M(8,8) = Act_inv(2,2);

        return M;
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
    void ComputeStretchingMatrix(FEM_Scaler lambda,FEM_Scaler mu,const Vec3d& Is,const Vec3d& sigma,Mat3x3d& A) const {
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
    
    Mat3x3d Qs[9];
};
