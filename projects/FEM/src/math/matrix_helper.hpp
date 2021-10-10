#ifndef EIGEN_HELPER_HPP
#define EIGEN_HELPER_HPP

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <Eigen/BlockSparseMatrix.h>
#include <vector>
#include <iostream>

#define FLOATING_PRECISION 2

#if FLOATING_PRECISION == 2
    #define FEM_Scaler double
    #define asFEMScaler asDouble
    #define GL_SCALER GL_DOUBLE
#else
    #define FEM_Scaler float
    #define asFEMScaler asFloat
    #define GL_SCALER GL_FLOAT
#endif 

#define SPMAT_SCALER FEM_Scaler

typedef Eigen::Matrix<FEM_Scaler,4,4,Eigen::RowMajor> Mat4x4d;
typedef Eigen::Matrix<FEM_Scaler,3,3,Eigen::RowMajor> Mat3x3d;
typedef Eigen::Matrix<FEM_Scaler,2,2,Eigen::RowMajor> Mat2x2d;
typedef Eigen::Matrix<FEM_Scaler,9,12,Eigen::RowMajor> Mat9x12d;        // deformation gradient mapping
typedef Eigen::Matrix<FEM_Scaler,12,12,Eigen::RowMajor> Mat12x12d;      // elm hessian
typedef Eigen::Matrix<FEM_Scaler,9,9,Eigen::RowMajor> Mat9x9d;
typedef Eigen::Matrix<FEM_Scaler,6,6,Eigen::RowMajor> Mat6x6d;          // stiffness tensor
typedef Eigen::Matrix<FEM_Scaler,6,12,Eigen::RowMajor> Mat6x12d;        // strain mapping tensor
typedef Eigen::Matrix<FEM_Scaler,24,24,Eigen::RowMajor> Mat24x24d;      // res jacobi tensor
typedef Eigen::Matrix<FEM_Scaler,9,3,Eigen::RowMajor> Mat9x3d;          // scaling space projection matrix

typedef Eigen::Matrix<FEM_Scaler,6,1> Vec6d;            

typedef Eigen::Matrix<FEM_Scaler, 9, Eigen::Dynamic, Eigen::RowMajor> Mat9xXd;
typedef Eigen::Matrix<FEM_Scaler, 12, Eigen::Dynamic, Eigen::RowMajor> Mat12xXd;

typedef Eigen::SparseMatrix<SPMAT_SCALER,Eigen::RowMajor>  SpMat;

typedef Eigen::Matrix<FEM_Scaler,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatXd;

typedef Eigen::Matrix<FEM_Scaler,2,1> Vec2d;
typedef Eigen::Matrix<FEM_Scaler,3,1> Vec3d;
typedef Eigen::Matrix<FEM_Scaler,4,1> Vec4d;
typedef Eigen::Matrix<FEM_Scaler,6,1> Vector6d;
typedef Eigen::Matrix<FEM_Scaler,12,1> Vec12d;
typedef Eigen::Matrix<FEM_Scaler,24,1> Vec24d;
typedef Eigen::Matrix<SPMAT_SCALER,Eigen::Dynamic,1> VecXd;
typedef Eigen::Matrix<int,9,1> Vec9i;
typedef Eigen::Matrix<FEM_Scaler,9,1> Vec9d;
typedef Eigen::Matrix<int,12,12> Mat12x12i;
typedef Eigen::Matrix<int,24,24> Mat24x24i;

typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::VectorXi VecXi;

typedef Eigen::Triplet<SPMAT_SCALER> Triplet;
typedef struct triplet_cmp{
    bool operator()(const Triplet &t1,const Triplet &t2) const{
        if(t1.row() < t2.row())
            return true;
        if(t1.row() == t2.row() && t1.col() < t2.col())
            return true;
        return false;
    }
} triplet_cmp;

class MatHelper{
public:
    static void UpdateDoFs(const SPMAT_SCALER* _src,SPMAT_SCALER* _des,int nm_updofs,const int *updofs){
        for(int i = 0;i < nm_updofs;++i){
            _des[updofs[i]] = _src[i];
        }
    }

     static void UpdateDoFs(FEM_Scaler up_value,SPMAT_SCALER* _des,int nm_updofs,const int *updofs){
        for(int i = 0;i < nm_updofs;++i){
            _des[updofs[i]] = up_value;
        }
    }   
    static void RetrieveDoFs(const SPMAT_SCALER* _src, SPMAT_SCALER* _des,int nm_rtdofs,const int *rtdofs){
        for(int i = 0;i < nm_rtdofs;++i)
            _des[i] = _src[rtdofs[i]];
    }
    static void RetrieveDoFs(const SPMAT_SCALER* from,SPMAT_SCALER* to,int nm_rtdofs,const int *rtdofs,int width){
        for(int i = 0;i < nm_rtdofs;++i)
            memcpy(&to[i*width],&from[rtdofs[i]*width],sizeof(SPMAT_SCALER)*width);  
    }
    static void AssembleSpMatAdd(const SpMat& src,SpMat &des,int row_offset,int col_offset,FEM_Scaler scale = 1.0){
        for(int k =0;k < int(src.outerSize());++k){
            for(SpMat::InnerIterator it(src,k);it;++it){
                int row = int(it.row());
                int col = int(it.col());
                des.coeffRef(row + row_offset,col + col_offset) += it.value()*scale;
            }
        }
    }

    static Vec9d VEC(const Mat3x3d& T2){
        Vec9d vec;
        vec << T2(0,0),T2(1,0),T2(2,0),T2(0,1),T2(1,1),T2(2,1),T2(0,2),T2(1,2),T2(2,2);
        return vec;
    }

    static Mat9x9d VEC(const std::vector<Mat3x3d> T4){
        Mat9x9d mat;
        for(int i = 0;i < 9;++i)
            mat.col(i) = VEC(T4[i]);

        return mat;
    }

    static Mat3x3d MAT(const Vec9d& vec) {
        Mat3x3d mat;
        mat << vec[0], vec[3], vec[6], vec[1], vec[4], vec[7], vec[2], vec[5], vec[8];
        return mat;
    }

    static Mat3x3d ASYM(const Vec3d& v){
        Mat3x3d C;
        C <<    0,-v[2],v[1],
                v[2],0,-v[0],
                -v[1],v[0],0;
        return C;
    }

    static Mat3x3d DYADIC(const Vec3d& v1,const Vec3d& v2){
        Mat3x3d C;
        C.setZero();
        for(int i = 0;i < 3;++i)
            for(int j = 0;j < 3;++j)
                C(i,j) = v1[i] * v2[j];

        return C;
    }

    static Mat9x9d DYADIC(const Vec9d& v1,const Vec9d& v2){
        Mat9x9d C;
        C.setZero();
        for(int i = 0;i < 9;++i)
            for(int j = 0;j < 9;++j)
                C(i,j) = v1[i] * v2[j];

        return C;
    }

};



#endif