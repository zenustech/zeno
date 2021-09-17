#pragma once

#include <matrix_helper.hpp>
#include <Eigen/Jacobi>
/**
 * @class <DiffSVD>
 * @brief Performing SVD or some other relavent decomposition.
 * 
 * Detail for SVD: For any 3x3 matrix F, we can compute its SVD F = U*Diag(s)*V', with both U and V othogonal matrix and Diag(s) a diagonal matrix. We call s
 * as the singular value of F. There are than one ways to define U,s,and V, in our implementation, different from the standard SVD in Eigen, the orthogonal matrices U and V
 * in our impletation are guaranteed to be pure rotation matrix without reflection, i.e U.determinant() = V.determinant() = 1, further more, we place the three singular values
 * in a descending order in s, and in order to maintain the reflection-free property of U and V, the smallest singular value(i.e s[2]) may be negative.
 * 
 * The class provides polar decomposition functionality, with F = R*S where R is in our implemention a pure rotation matrix and S a symmtric matrix. The polar decomposition 
 * is direcly implemented over the SVD of this class, with R = U * V and S = V' * Diag(s) * V.
 * 
 * The class also provide the gradient computation and SVD decomposition(thus I name the class DiffSVD) w.r.t the entry of F. This is mainly for the computation of the gradient
 * of First Piola Stress w.r.t the deformation gradient. Take a deformation gradient F with SVD F = U * Diag(s) * V', for isotropic model, the first Piola stress can be defined by
 * P = U * Diag(p) * V' and p = p(s) which is defined by specific isotropic hyper elastic force model. We provide the computation of the gradient of P w.r.t (i,j) entry of F, i.e
 * (dP)/(dF(i,j)), and user has to provide the SVD of F as well as the principal stress p and tha gradient of p over s(a 3x3 matrix) which are defined by specific force model as the input.
 */
class DiffSVD
{
private:
    friend class SVD_UNITTEST;
    /**
     * @brief Constructor of SVD decomposition solver.
     * @param epsilon default to be 1e-6, if the difference of two singular values is smaller than this threshold, the algorithm will consider the two singular values to be identical,
     * mainly used in gradient decomposition of SVD decomposition.
     */
    DiffSVD() {}
    /**
     * @brief A destructor which does nothing.
     */
    ~DiffSVD() {}

public:
    /**
     * @brief The eigen value decomposition of a 3x3 symmetric matrix.
     * @param sym_A A symmetric matrix for eigen decomposition, if the input matrix is not symmetric, the function will return the eigen decomposition of 0.5 * (A + A').
     * @param eigen_vals the eigen values of symmetric matrix.
     * @param eigen_vecs the eigen vectors of symmtrix matrix which satisfy A * eigen_vecs[i] = eigen_vals[i] * eigen_vecs[i].
     * @return determine whether the decomposition if successful, currently the function will always return 0.
     */
    static int SYM_Eigen_Decomposition(const Mat3x3d& sym_A,Vec3d& eigen_vals,Mat3x3d& eigen_vecs);
    /**
     * @brief The singular value decomposition of a 3x3 matrix F = U * diag(sigma) * V'.
     * @param elmF A 3x3 matrix for singular value decomposition
     * @param elmU the U matrix of singular value decomposition, and elmU guarantees to be a rotation only matrix without reflection, which means U.determinant() = 1
     * @param elmSigma the singular values matrix elmF, with singular values in a descending order. elmSigma[2] might be a negative value because we want to make sure
     * matrix U and V are reflection-free. 
     * @param elmV the V matrix of singular value decomposition, and elmV guarantees to be a rotation only matrix without reflection, which means V.determinant() = 1.
     * @return determine whether the decomposition if successful, currently the function will always return 0.
     */
    static int SVD_Decomposition(const Mat3x3d& elmF, Mat3x3d& elmU, Vec3d& elmSigma, Mat3x3d& elmV);
    /**
     * @brief The polar decomposition of a 3x3 matrix F = R * S, where R is a reflection-free rotation matrix, and S is a symmetric matrix and might contain reflection.
     * @param elmF A 3x3 matrix for polar decomposition
     * @param elmR The 3x3 rotation matrix of polar decomposition of elmF
     * @param elmS the 3x3 symmetric matrix of polar decomposition of elmF
     */
    static int Polar_Decomposition(const Mat3x3d& elmF, Mat3x3d& elmR, Mat3x3d& elmS);
    /**
     * @brief Given the polar decompsition of F = R * S, as well as the two rotation matrices U and V of singular value decomposition F = U * S * V',
     * the function compute the gradient of rotation matrix R w.r.t F(i,j)
     * @param i Specify row index of F matrix in computing the gradient of rotation matrix R w.r.t F(i,j)
     * @param j Specify the col index of F matrix in computing the gradient of rotation matrix R w.r.t F(i,j)
     * @param U the reflection-free rotation matrix U of singular value decomposition of F = U * S * V'
     * @param V the reflection-free rotation matrix V of singular value decomposition of F = U * S * V'
     * @param lambda the singular values of matrix F, and lambda[2] might be negative.
     * @param dR the gradient of matrix R w.r.t F(i,j)
     */
    static void Compute_dRdF(size_t i,size_t j,const Mat3x3d& U,const Mat3x3d& V,const Vec3d& lambda,Mat3x3d& dR);
    /**
     * @brief Given the singular value decomposition of the deformation gradient matrix F = U * Diag(s) * V', together with the principal stress phat and dphat/ds,
     * this function compute the gradient of first-Piola stress P = U * phat V' w.r.t F(i,j).
     * @param i Specify row index of F matrix in computing the gradient of first-Piola stress P w.r.t F(i,j).
     * @param j Specify the col index of F matrix in computing the gradient of first-Piola stress P w.r.t F(i,j).
     * @param U the reflection-free rotation matrix U of singular value decomposition of F = U * S * V'
     * @param V the reflection-free rotation matrix V of singular value decomposition of F = U * S * V'
     * @param sigma the singular values of F, i.e the principal stretches
     * @param phat the principal stresses of the principal stretches sigma
     * @param dpds the gradient of principal stresses phat w.r.t the principal stretches sigma.
     * @param dP the gradient of first Piola stress P = U * phat V' w.r.t F(i,j)
     */
    static void Compute_dPdF(size_t i,size_t j,const Mat3x3d& U,
                                  const Mat3x3d& V,
                                  const Vec3d& sigma,//principal strain
                                  const Vec3d& phat,//principal stress
                                  const Mat3x3d& dpds,//derivative of 'p'hat with repect to 's'igma
                                  Mat3x3d& dP);
private:
    static const FEM_Scaler epsilon;/** threshold on determining whether two singular values are the same.*/
};

