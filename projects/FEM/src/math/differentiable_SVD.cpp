#include "differentiable_SVD.h"
#include <iostream>

const FEM_Scaler DiffSVD::epsilon = 1e-6;

int DiffSVD::SVD_Decomposition(const Mat3x3d& F,Mat3x3d& U, Vec3d& Sigma,Mat3x3d& V) {
    Eigen::JacobiSVD<Mat3x3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();
    Sigma = svd.singularValues();

    FEM_Scaler U_det = U.determinant();
    FEM_Scaler V_det = V.determinant();
    FEM_Scaler L_det = U_det * V_det;

    if(U_det < 0 && V_det > 0)
        U.col(2) *= L_det;
    else if(U_det > 0 && V_det < 0)
        V.col(2) *= L_det;

    Sigma(2) *= L_det;

    return 0;
}

int DiffSVD::SYM_Eigen_Decomposition(const Mat3x3d& _sym_A,Vec3d& eigen_vals,Mat3x3d& eigen_vecs) {
    Mat3x3d sym_A = 0.5 * (_sym_A.transpose() + _sym_A);

    Eigen::JacobiSVD<Mat3x3d> svd(sym_A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3x3d U,V;
    U = svd.matrixU();
    V = svd.matrixV();
    eigen_vecs = U;
    eigen_vals = svd.singularValues();
    Mat3x3d L = V.transpose() * U;

    eigen_vecs = eigen_vecs * L;
    eigen_vals = L * eigen_vals;

    return 0;
}

int DiffSVD::Polar_Decomposition(const Mat3x3d& elm_F, Mat3x3d& elm_R, Mat3x3d& elm_S) {
    Mat3x3d U,V;
    Vec3d sigma;
    int state = SVD_Decomposition(elm_F,U,sigma,V);
    elm_R = U * V.transpose();
    elm_S = V * sigma.asDiagonal() * V.transpose();
    return state;
}

void DiffSVD::Compute_dRdF(size_t k,size_t l,const Mat3x3d& U,const Mat3x3d& V,const Vec3d& sigma,Mat3x3d& dR) {
    Mat3x3d dF = Mat3x3d::Zero();
    dF(k,l) = 1.0;
    Mat3x3d W = U.transpose() * dF * V;
    dR.setZero();
    for(size_t i = 0;i < 3;++i)                                                                                                                                                                                                                                                                                                                                                    
        for(size_t j = i+1;j < 3;++j){
            if(i == j)
                continue;
            FEM_Scaler wij = W(i,j);
            FEM_Scaler wji = W(j,i);
            FEM_Scaler si = sigma[i];
            FEM_Scaler sj = sigma[j];
            dR(i,j) = (wij - wji)/(si + sj);
            dR(j,i) = -dR(i,j);
        }

    dR = U * dR * V.transpose(); 
}

void DiffSVD::Compute_dPdF(size_t k,size_t l,const Mat3x3d& U,
                                const Mat3x3d& V,
                                const Vec3d& sigma,
                                const Vec3d& phat,
                                const Mat3x3d& dpds,
                                Mat3x3d& dP) {
    Mat3x3d dF = Mat3x3d::Zero();
    dF(k,l) = 1.0;
    Vec3d ds = (U.transpose() * dF * V).diagonal();
    Vec3d dp = dpds * ds;
    Mat3x3d W = U.transpose() * dF * V;
    dP.setZero();
    dP.diagonal() = dp;
    for(size_t i = 0;i < 3;++i)
        for(size_t j = 0;j < 3;++j){
            if(i == j)
                continue;
            FEM_Scaler wij = W(i,j);
            FEM_Scaler wji = W(j,i);
            FEM_Scaler si = sigma[i];
            FEM_Scaler sj = sigma[j];
            FEM_Scaler pi = phat[i];
            FEM_Scaler pj = phat[j];
            if(fabs(sigma[i] - sigma[j]) < epsilon){
                FEM_Scaler psij = dpds(i,j);
                FEM_Scaler psjj = dpds(j,j);
                // dP(i,j) = (si*wij + sj*wji)*psjj + (wji*pj - wij*pi) - (wij*sj + wji*si)*psij;
                dP(i,j) = (psjj*sj + pj - psij*si)*wij + (si*psjj - pi - sj*psij)*wji;
                dP(i,j) /= (sj + si);
            }else{
                dP(i,j) = (sj*pj - si*pi)*wij + (si*pj - sj*pi)*wji;
                dP(i,j) /= (sj+si)*(sj-si);
            }
        }
    dP = U * dP * V.transpose();    
}
