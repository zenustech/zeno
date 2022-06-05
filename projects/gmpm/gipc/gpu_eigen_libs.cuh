#pragma once
#ifndef _GPU_EIGEN_LIB_CUH_
#define _GPU_EIGEN_LIB_CUH_
#include <cuda_runtime.h>
#include "eigen_data.h"
#include "svd.cuh"

__device__
double atomicAdd_double(double* address, double val);

namespace __GEIGEN__ {
	__device__ __host__ void __init_Mat3x3(Matrix3x3d& M, const double& val);

	__device__ __host__ void __init_Mat6x6(Matrix6x6d& M, const double& val);

	__device__ __host__ void __init_Mat9x9(Matrix9x9d& M, const double& val);

	__device__ __host__ void __identify_Mat9x9(Matrix9x9d& M);

	__device__ __host__ double __mabs(const double& a);

	__device__ __host__ double __norm(const double3& n);

	__device__ __host__ double3 __s_vec_multiply(const double3& a, double b);

	__device__ __host__ double2 __s_vec_multiply(const double2& a, double b);

	__device__ __host__ double3 __normalized(double3 n);

	__device__ __host__ double3 __add(double3 a, double3 b);

	__device__ __host__ Vector9 __add9(const Vector9& a, const Vector9& b);

	__device__ __host__ double3 __minus(double3 a, double3 b);

	__device__ __host__ double2 __minus_v2(double2 a, double2 b);

	__device__ __host__ double3 __v_vec_multiply(double3 a, double3 b);

	__device__ __host__ double __squaredNorm(double3 a);

	__device__ __host__ double __squaredNorm(double2 a);

	__device__ __host__ void __M_Mat_multiply(const Matrix3x3d& A, const Matrix3x3d& B, Matrix3x3d& output);

	__device__ __host__ Matrix3x3d __M_Mat_multiply(const Matrix3x3d& A, const Matrix3x3d& B);

	__device__ __host__ Matrix2x2d __M2x2_Mat2x2_multiply(const Matrix2x2d& A, const Matrix2x2d& B);

	__device__ __host__ double __Mat_Trace(const Matrix3x3d& A);

	__device__ __host__ double3 __v_M_multiply(const double3& n, const Matrix3x3d& A);

	__device__ __host__ double3 __M_v_multiply(const Matrix3x3d& A, const double3& n);

	__device__ __host__ double3 __M3x2_v2_multiply(const Matrix3x2d& A, const double2& n);

	__device__ __host__ Vector12 __M12x9_v9_multiply(const Matrix12x9d& A, const Vector9& n);

	__device__ __host__ Vector12 __M12x6_v6_multiply(const Matrix12x6d& A, const Vector6& n);

	__device__ __host__ Vector6 __M6x3_v3_multiply(const Matrix6x3d& A, const double3& n);

	__device__ __host__ double2 __M2x3_v3_multiply(const Matrix2x3d& A, const double3& n);

	__device__ __host__ Vector9 __M9x6_v6_multiply(const Matrix9x6d& A, const Vector6& n);

	__device__ __host__ Vector12 __M12x12_v12_multiply(const Matrix12x12d& A, const Vector12& n);

	__device__ __host__ Vector9 __M9x9_v9_multiply(const Matrix9x9d& A, const Vector9& n);

	__device__ __host__ Vector6 __M6x6_v6_multiply(const Matrix6x6d& A, const Vector6& n);

	__device__ __host__ Matrix9x9d __S_Mat9x9_multiply(const Matrix9x9d& A, const double& B);

	__device__ __host__ Matrix6x6d __S_Mat6x6_multiply(const Matrix6x6d& A, const double& B);

	__device__ __host__ double __v_vec_dot(const double3& a, const double3& b);

	__device__ __host__ double3 __v_vec_cross(double3 a, double3 b);

	__device__ __host__ Matrix3x3d __v_vec_toMat(double3 a, double3 b);

	__device__ __host__ Matrix2x2d __v2_vec2_toMat2x2(double2 a, double2 b);

	__device__ __host__ Matrix2x2d __s_Mat2x2_multiply(Matrix2x2d A, double b);

	__device__ __host__ Matrix2x2d __Mat2x2_minus(Matrix2x2d A, Matrix2x2d B);

	__device__ __host__ Matrix3x3d __Mat3x3_minus(Matrix3x3d A, Matrix3x3d B);

	__device__ __host__ Matrix9x9d __v9_vec9_toMat9x9(Vector9 a, Vector9 b);

	__device__ __host__ Matrix6x6d __v6_vec6_toMat6x6(Vector6 a, Vector6 b);

	__device__ __host__ Vector9 __s_vec9_multiply(Vector9 a, double b);

	__device__ __host__ Vector12 __s_vec12_multiply(Vector12 a, double b);

	__device__ __host__ Vector6 __s_vec6_multiply(Vector6 a, double b);

	__device__ __host__ void __Mat_add(const Matrix3x3d& A, const Matrix3x3d& B, Matrix3x3d& output);

	__device__ __host__ Matrix3x3d __Mat_add(const Matrix3x3d& A, const Matrix3x3d& B);

	__device__ __host__ Matrix2x2d __Mat2x2_add(const Matrix2x2d& A, const Matrix2x2d& B);

	__device__ __host__ Matrix9x9d __Mat9x9_add(const Matrix9x9d& A, const Matrix9x9d& B);

	__device__ __host__ Matrix9x12d __Mat9x12_add(const Matrix9x12d& A, const Matrix9x12d& B);

	__device__ __host__ Matrix6x12d __Mat6x12_add(const Matrix6x12d& A, const Matrix6x12d& B);

	__device__ __host__ Matrix6x9d __Mat6x9_add(const Matrix6x9d& A, const Matrix6x9d& B);

	__device__ __host__ Matrix3x6d __Mat3x6_add(const Matrix3x6d& A, const Matrix3x6d& B);

	__device__ __host__ void __set_Mat_identity(Matrix2x2d& M);

	__device__ __host__ void __set_Mat_val(Matrix3x3d& M, const double& a00, const double& a01, const double& a02,
		const double& a10, const double& a11, const double& a12,
		const double& a20, const double& a21, const double& a22);

	__device__ __host__ void __set_Mat_val_row(Matrix3x3d& M, const double3& row0, const double3& row1, const double3& row2);

	__device__ __host__ void __set_Mat_val_column(Matrix3x3d& M, const double3& col0, const double3& col1, const double3& col2);

	__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2d& M, const double3& col0, const double3& col1);

	__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2d& M, const double2& col0, const double2& col1);

	__device__ __host__ void __init_Mat9x12_val(Matrix9x12d& M, const double& val);

	__device__ __host__ void __init_Mat6x12_val(Matrix6x12d& M, const double& val);

	__device__ __host__ void __init_Mat6x9_val(Matrix6x9d& M, const double& val);

	__device__ __host__ void __init_Mat3x6_val(Matrix3x6d& M, const double& val);

	__device__ __host__ Matrix3x3d __S_Mat_multiply(const Matrix3x3d& A, const double& B);



	__device__ __host__ Matrix3x3d __Transpose3x3(Matrix3x3d input);

	__device__ __host__ Matrix12x9d __Transpose9x12(const Matrix9x12d& input);

	__device__ __host__ Matrix2x3d __Transpose3x2(const Matrix3x2d& input);

	__device__ __host__ Matrix9x12d __Transpose12x9(const Matrix12x9d& input);

	__device__ __host__ Matrix12x6d __Transpose6x12(const Matrix6x12d& input);

	__device__ __host__ Matrix9x6d __Transpose6x9(const Matrix6x9d& input);

	__device__ __host__ Matrix6x3d __Transpose3x6(const Matrix3x6d& input);

	__device__ __host__ Matrix12x9d __M12x9_M9x9_Multiply(const Matrix12x9d& A, const Matrix9x9d& B);

	__device__ __host__ Matrix12x6d __M12x6_M6x6_Multiply(const Matrix12x6d& A, const Matrix6x6d& B);

	__device__ __host__ Matrix9x6d __M9x6_M6x6_Multiply(const Matrix9x6d& A, const Matrix6x6d& B);

	__device__ __host__ Matrix6x3d __M6x3_M3x3_Multiply(const Matrix6x3d& A, const Matrix3x3d& B);

	__device__ __host__ Matrix3x2d __M3x2_M2x2_Multiply(const Matrix3x2d& A, const Matrix2x2d& B);

	__device__ __host__ Matrix12x12d __M12x9_M9x12_Multiply(const Matrix12x9d& A, const Matrix9x12d& B);

	__device__ __host__ Matrix12x2d __M12x2_M2x2_Multiply(const Matrix12x2d& A, const Matrix2x2d& B);

	__device__ __host__ Matrix9x2d __M9x2_M2x2_Multiply(const Matrix9x2d& A, const Matrix2x2d& B);

	__device__ __host__ Matrix6x2d __M6x2_M2x2_Multiply(const Matrix6x2d& A, const Matrix2x2d& B);

	__device__ __host__ Matrix12x12d __M12x2_M12x2T_Multiply(const Matrix12x2d& A, const Matrix12x2d& B);

	__device__ __host__ Matrix9x9d __M9x2_M9x2T_Multiply(const Matrix9x2d& A, const Matrix9x2d& B);

	__device__ __host__ Matrix6x6d __M6x2_M6x2T_Multiply(const Matrix6x2d& A, const Matrix6x2d& B);

	__device__ __host__ Matrix12x12d __M12x6_M6x12_Multiply(const Matrix12x6d& A, const Matrix6x12d& B);

	__device__ __host__ Matrix9x9d __M9x6_M6x9_Multiply(const Matrix9x6d& A, const Matrix6x9d& B);

	__device__ __host__ Matrix6x6d __M6x3_M3x6_Multiply(const Matrix6x3d& A, const Matrix3x6d& B);

	__device__ __host__ Matrix12x12d __s_M12x12_Multiply(const Matrix12x12d& A, const double& B);

	__device__ __host__ Matrix9x9d __s_M9x9_Multiply(const Matrix9x9d& A, const double& B);

	__device__ __host__ Matrix6x6d __s_M6x6_Multiply(const Matrix6x6d& A, const double& B);

	__device__ __host__ void __Determiant(const Matrix3x3d& input, double& determinant);

	__device__ __host__ double __Determiant(const Matrix3x3d& input);

	__device__ __host__ void __Inverse(const Matrix3x3d& input, Matrix3x3d& output);

	__device__ __host__ void __Inverse2x2(const Matrix2x2d& input, Matrix2x2d& output);

	__device__ __host__ double __f(const double& x, const double& a, const double& b, const double& c, const double& d);

	__device__ __host__ double __df(const double& x, const double& a, const double& b, const double& c);

	__device__ __host__ void __NewtonSolverForCubicEquation(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS = 1e-6);

	__device__ __host__ void __SolverForCubicEquation(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS = 1e-6);

	__device__ __host__ Vector9 __Mat3x3_to_vec9_double(const Matrix3x3d& F);

	__device__ __host__ void __normalized_vec9_double(Vector9& v9);


	__device__ __host__ Vector6 __Mat3x2_to_vec6_double(const Matrix3x2d& F);

	__device__ __host__ Matrix3x3d __vec9_to_Mat3x3_double(const double vec9[9]);

	__device__ __host__ Matrix2x2d __vec4_to_Mat2x2_double(const double vec4[4]);

	__device__ void SVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma);

	__device__ __host__ void __makePD2x2(const double& a00, const double& a01, const double& a10, const double& a11, double eigenValues[2], int& num, double2 eigenVectors[2], double eps = 1e-32);
}

#endif //_GPU_EIGEN_LIB_CUH_