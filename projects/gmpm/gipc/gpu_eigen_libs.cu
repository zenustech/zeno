#pragma once
#include "gpu_eigen_libs.cuh"
#include "math.h"

__device__
double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}


namespace __GEIGEN__ {

	__device__ __host__ void __init_Mat3x3(Matrix3x3d& M, const double& val) {
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 3;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat6x6(Matrix6x6d& M, const double& val) {
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 6;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat9x9(Matrix9x9d& M, const double& val) {
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __identify_Mat9x9(Matrix9x9d& M) {
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				if (i == j) {
					M.m[i][j] = 1;
				}
				else {
					M.m[i][j] = 0;
				}
			}
		}
	}

	__device__ __host__ double __mabs(const double& a) {
		return a > 0 ? a : -a;
	}

	__device__ __host__ double __norm(const double3& n) {
		return sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
	}

	__device__ __host__ double3 __s_vec_multiply(const double3& a, double b) {
		return make_double3(a.x * b, a.y * b, a.z * b);
	}

	__device__ __host__ double2 __s_vec_multiply(const double2& a, double b) {
		return make_double2(a.x * b, a.y * b);
	}

	__device__ __host__ double3 __normalized(double3 n) {
		double norm = __norm(n);
		norm = 1 / norm;
		return __s_vec_multiply(n, norm);
	}

	__device__ __host__ double3 __add(double3 a, double3 b) {
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	__device__ __host__ Vector9 __add9(const Vector9& a, const Vector9& b) {
		Vector9 V;
		for (int i = 0;i < 9;i++) {
			V.v[i] = a.v[i] + b.v[i];
		}
		return V;
	}

	__device__ __host__ double3 __minus(double3 a, double3 b) {
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	__device__ __host__ double2 __minus_v2(double2 a, double2 b) {
		return make_double2(a.x - b.x, a.y - b.y);
	}

	__device__ __host__ double3 __v_vec_multiply(double3 a, double3 b) {
		return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
	}

	__device__ __host__ double __squaredNorm(double3 a) {
		return a.x * a.x + a.y * a.y + a.z * a.z;
	}

	__device__ __host__ double __squaredNorm(double2 a)
	{
		return  a.x * a.x + a.y * a.y;
	}

	__device__ __host__ void __M_Mat_multiply(const Matrix3x3d& A, const Matrix3x3d& B, Matrix3x3d& output) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
	}


	__device__ __host__ Matrix3x3d __M_Mat_multiply(const Matrix3x3d& A, const Matrix3x3d& B) {
		Matrix3x3d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix2x2d __M2x2_Mat2x2_multiply(const Matrix2x2d& A, const Matrix2x2d& B) {
		Matrix2x2d output;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ double __Mat_Trace(const Matrix3x3d& A) {
		return A.m[0][0] + A.m[1][1] + A.m[2][2];
	}

	__device__ __host__ double3 __v_M_multiply(const double3& n, const Matrix3x3d& A) {
		double x = A.m[0][0] * n.x + A.m[1][0] * n.y + A.m[2][0] * n.z;
		double y = A.m[0][1] * n.x + A.m[1][1] * n.y + A.m[2][1] * n.z;
		double z = A.m[0][2] * n.x + A.m[1][2] * n.y + A.m[2][2] * n.z;
		return make_double3(x, y, z);
	}

	__device__ __host__ double3 __M_v_multiply(const Matrix3x3d& A, const double3& n) {
		double x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
		double y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
		double z = A.m[2][0] * n.x + A.m[2][1] * n.y + A.m[2][2] * n.z;
		return make_double3(x, y, z);
	}

	__device__ __host__ double3 __M3x2_v2_multiply(const Matrix3x2d& A, const double2& n) {
		double x = A.m[0][0] * n.x + A.m[0][1] * n.y;// +A.m[0][2] * n.z;
		double y = A.m[1][0] * n.x + A.m[1][1] * n.y;// +A.m[1][2] * n.z;
		double z = A.m[2][0] * n.x + A.m[2][1] * n.y;// +A.m[2][2] * n.z;
		return make_double3(x, y, z);
	}



	__device__ __host__ Vector12 __M12x9_v9_multiply(const Matrix12x9d& A, const Vector9& n) {
		Vector12 v12;
		for (int i = 0;i < 12;i++) {
			double temp = 0;
			for (int j = 0;j < 9;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v12.v[i] = temp;
		}
		return v12;
	}

	__device__ __host__ Vector12 __M12x6_v6_multiply(const Matrix12x6d& A, const Vector6& n) {
		Vector12 v12;
		for (int i = 0;i < 12;i++) {
			double temp = 0;
			for (int j = 0;j < 6;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v12.v[i] = temp;
		}
		return v12;
	}

	__device__ __host__ Vector6 __M6x3_v3_multiply(const Matrix6x3d& A, const double3& n) {
		Vector6 v6;
		for (int i = 0;i < 6;i++) {

			double temp = A.m[i][0] * n.x;
			temp += A.m[i][1] * n.y;
			temp += A.m[i][2] * n.z;

			v6.v[i] = temp;
		}
		return v6;
	}

	__device__ __host__ double2 __M2x3_v3_multiply(const Matrix2x3d& A, const double3& n) {
		double2 output;
		output.x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
		output.y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
		return output;
	}

	__device__ __host__ Vector9 __M9x6_v6_multiply(const Matrix9x6d& A, const Vector6& n) {
		Vector9 v9;
		for (int i = 0;i < 9;i++) {
			double temp = 0;
			for (int j = 0;j < 6;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v9.v[i] = temp;
		}
		return v9;
	}

	__device__ __host__ Vector12 __M12x12_v12_multiply(const Matrix12x12d& A, const Vector12& n) {
		Vector12 v12;
		for (int i = 0;i < 12;i++) {
			double temp = 0;
			for (int j = 0;j < 12;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v12.v[i] = temp;
		}
		return v12;
	}

	__device__ __host__ Vector9 __M9x9_v9_multiply(const Matrix9x9d& A, const Vector9& n) {
		Vector9 v9;
		for (int i = 0;i < 9;i++) {
			double temp = 0;
			for (int j = 0;j < 9;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v9.v[i] = temp;
		}
		return v9;
	}

	__device__ __host__ Vector6 __M6x6_v6_multiply(const Matrix6x6d& A, const Vector6& n) {
		Vector6 v6;
		for (int i = 0;i < 6;i++) {
			double temp = 0;
			for (int j = 0;j < 6;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v6.v[i] = temp;
		}
		return v6;
	}

	__device__ __host__ Matrix9x9d __S_Mat9x9_multiply(const Matrix9x9d& A, const double& B) {
		Matrix9x9d output;
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				output.m[i][j] = A.m[i][j] * B;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x6d __S_Mat6x6_multiply(const Matrix6x6d& A, const double& B) {
		Matrix6x6d output;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 6;j++) {
				output.m[i][j] = A.m[i][j] * B;
			}
		}
		return output;
	}

	__device__ __host__ double __v_vec_dot(const double3& a, const double3& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __host__ double3 __v_vec_cross(double3 a, double3 b) {
		return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}

	__device__ __host__ Matrix3x3d __v_vec_toMat(double3 a, double3 b) {
		Matrix3x3d M;
		M.m[0][0] = a.x * b.x;M.m[0][1] = a.x * b.y;M.m[0][2] = a.x * b.z;
		M.m[1][0] = a.y * b.x;M.m[1][1] = a.y * b.y;M.m[1][2] = a.y * b.z;
		M.m[2][0] = a.z * b.x;M.m[2][1] = a.z * b.y;M.m[2][2] = a.z * b.z;
		return M;
	}

	__device__ __host__ Matrix2x2d __v2_vec2_toMat2x2(double2 a, double2 b) {
		Matrix2x2d M;
		M.m[0][0] = a.x * b.x;M.m[0][1] = a.x * b.y;
		M.m[1][0] = a.y * b.x;M.m[1][1] = a.y * b.y;
		return M;
	}

	__device__ __host__ Matrix2x2d __s_Mat2x2_multiply(Matrix2x2d A, double b)
	{
		Matrix2x2d output;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[i][j] = A.m[i][j] * b;
			}
		}
		return output;
	}

	__device__ __host__ Matrix2x2d __Mat2x2_minus(Matrix2x2d A, Matrix2x2d B)
	{
		Matrix2x2d output;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[i][j] = A.m[i][j] - B.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix3x3d __Mat3x3_minus(Matrix3x3d A, Matrix3x3d B)
	{
		Matrix3x3d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				output.m[i][j] = A.m[i][j] - B.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x9d __v9_vec9_toMat9x9(Vector9 a, Vector9 b) {
		Matrix9x9d M;
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				M.m[i][j] = a.v[i] * b.v[j];
			}
		}
		return M;
	}

	__device__ __host__ Matrix6x6d __v6_vec6_toMat6x6(Vector6 a, Vector6 b) {
		Matrix6x6d M;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 6;j++) {
				M.m[i][j] = a.v[i] * b.v[j];
			}
		}
		return M;
	}

	__device__ __host__ Vector9 __s_vec9_multiply(Vector9 a, double b) {
		Vector9 V;
		for (int i = 0;i < 9;i++)
			V.v[i] = a.v[i] * b;
		return V;
	}

	__device__ __host__ Vector12 __s_vec12_multiply(Vector12 a, double b) {
		Vector12 V;
		for (int i = 0;i < 12;i++)
			V.v[i] = a.v[i] * b;
		return V;
	}

	__device__ __host__ Vector6 __s_vec6_multiply(Vector6 a, double b) {
		Vector6 V;
		for (int i = 0;i < 6;i++)
			V.v[i] = a.v[i] * b;
		return V;
	}

	__device__ __host__ void __Mat_add(const Matrix3x3d& A, const Matrix3x3d& B, Matrix3x3d& output) {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
	}

	__device__ __host__ Matrix3x3d __Mat_add(const Matrix3x3d& A, const Matrix3x3d& B) {
		Matrix3x3d output;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix2x2d __Mat2x2_add(const Matrix2x2d& A, const Matrix2x2d& B) {
		Matrix2x2d output;
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix9x9d __Mat9x9_add(const Matrix9x9d& A, const Matrix9x9d& B) {
		Matrix9x9d output;
		for (int i = 0; i < 9; i++)
			for (int j = 0; j < 9; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix9x12d __Mat9x12_add(const Matrix9x12d& A, const Matrix9x12d& B) {
		Matrix9x12d output;
		for (int i = 0; i < 9; i++)
			for (int j = 0; j < 12; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix6x12d __Mat6x12_add(const Matrix6x12d& A, const Matrix6x12d& B) {
		Matrix6x12d output;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 12; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix6x9d __Mat6x9_add(const Matrix6x9d& A, const Matrix6x9d& B) {
		Matrix6x9d output;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 9; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix3x6d __Mat3x6_add(const Matrix3x6d& A, const Matrix3x6d& B) {
		Matrix3x6d output;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 6; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ void __set_Mat_identity(Matrix2x2d& M) {
		M.m[0][0] = 1;
		M.m[1][0] = 0;
		M.m[0][1] = 0;
		M.m[1][1] = 1;
	}

	__device__ __host__ void __set_Mat_val(Matrix3x3d& M, const double& a00, const double& a01, const double& a02,
		const double& a10, const double& a11, const double& a12,
		const double& a20, const double& a21, const double& a22) {
		M.m[0][0] = a00;M.m[0][1] = a01;M.m[0][2] = a02;
		M.m[1][0] = a10;M.m[1][1] = a11;M.m[1][2] = a12;
		M.m[2][0] = a20;M.m[2][1] = a21;M.m[2][2] = a22;
	}

	__device__ __host__ void __set_Mat_val_row(Matrix3x3d& M, const double3& row0, const double3& row1, const double3& row2) {
		M.m[0][0] = row0.x;M.m[0][1] = row0.y;M.m[0][2] = row0.z;
		M.m[1][0] = row1.x;M.m[1][1] = row1.y;M.m[1][2] = row1.z;
		M.m[2][0] = row2.x;M.m[2][1] = row2.y;M.m[2][2] = row2.z;
	}

	__device__ __host__ void __set_Mat_val_column(Matrix3x3d& M, const double3& col0, const double3& col1, const double3& col2) {
		M.m[0][0] = col0.x;M.m[0][1] = col1.x;M.m[0][2] = col2.x;
		M.m[1][0] = col0.y;M.m[1][1] = col1.y;M.m[1][2] = col2.y;
		M.m[2][0] = col0.z;M.m[2][1] = col1.z;M.m[2][2] = col2.z;
	}

	__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2d& M, const double3& col0, const double3& col1) {
		M.m[0][0] = col0.x;M.m[0][1] = col1.x;
		M.m[1][0] = col0.y;M.m[1][1] = col1.y;
		M.m[2][0] = col0.z;M.m[2][1] = col1.z;
	}

	__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2d& M, const double2& col0, const double2& col1) {
		M.m[0][0] = col0.x;M.m[0][1] = col1.x;
		M.m[1][0] = col0.y;M.m[1][1] = col1.y;
	}

	__device__ __host__ void __init_Mat9x12_val(Matrix9x12d& M, const double& val) {
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 12;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat6x12_val(Matrix6x12d& M, const double& val) {
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 12;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat6x9_val(Matrix6x9d& M, const double& val) {
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 9;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat3x6_val(Matrix3x6d& M, const double& val) {
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 6;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ Matrix3x3d __S_Mat_multiply(const Matrix3x3d& A, const double& B) {
		Matrix3x3d output;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ Matrix3x3d __Transpose3x3(Matrix3x3d input) {
		Matrix3x3d output;
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 3;j++) {
				output.m[i][j] = input.m[j][i];
			}
		}
		return output;
		//output.m[0][0] = input.m[0][0];output.m[0][1] = input.m[1][0];output.m[0][2] = input.m[2][0];
		//output.m[1][0] = input.m[0][1];output.m[1][1] = input.m[1][1];output.m[1][2] = input.m[2][1];
		//output.m[2][0] = input.m[0][2];output.m[2][1] = input.m[1][2];output.m[2][2] = input.m[2][2];
	}

	__device__ __host__ Matrix12x9d __Transpose9x12(const Matrix9x12d& input) {
		Matrix12x9d output;
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 12;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix2x3d __Transpose3x2(const Matrix3x2d& input) {
		Matrix2x3d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x12d __Transpose12x9(const Matrix12x9d& input) {
		Matrix9x12d output;
		for (int i = 0;i < 12;i++) {
			for (int j = 0;j < 9;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x6d __Transpose6x12(const Matrix6x12d& input) {
		Matrix12x6d output;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 12;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x6d __Transpose6x9(const Matrix6x9d& input) {
		Matrix9x6d output;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 9;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x3d __Transpose3x6(const Matrix3x6d& input) {
		Matrix6x3d output;
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 6;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x9d __M12x9_M9x9_Multiply(const Matrix12x9d& A, const Matrix9x9d& B) {
		Matrix12x9d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 9; j++) {
				double temp = 0;
				for (int k = 0; k < 9; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x6d __M12x6_M6x6_Multiply(const Matrix12x6d& A, const Matrix6x6d& B) {
		Matrix12x6d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x6d __M9x6_M6x6_Multiply(const Matrix9x6d& A, const Matrix6x6d& B) {
		Matrix9x6d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x3d __M6x3_M3x3_Multiply(const Matrix6x3d& A, const Matrix3x3d& B) {
		Matrix6x3d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix3x2d __M3x2_M2x2_Multiply(const Matrix3x2d& A, const Matrix2x2d& B) {
		Matrix3x2d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __M12x9_M9x12_Multiply(const Matrix12x9d& A, const Matrix9x12d& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				double temp = 0;
				for (int k = 0; k < 9; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x2d __M12x2_M2x2_Multiply(const Matrix12x2d& A, const Matrix2x2d& B) {
		Matrix12x2d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x2d __M9x2_M2x2_Multiply(const Matrix9x2d& A, const Matrix2x2d& B)
	{
		Matrix9x2d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x2d __M6x2_M2x2_Multiply(const Matrix6x2d& A, const Matrix2x2d& B)
	{
		Matrix6x2d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __M12x2_M12x2T_Multiply(const Matrix12x2d& A, const Matrix12x2d& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[j][k];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x9d __M9x2_M9x2T_Multiply(const Matrix9x2d& A, const Matrix9x2d& B)
	{
		Matrix9x9d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[j][k];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x6d __M6x2_M6x2T_Multiply(const Matrix6x2d& A, const Matrix6x2d& B)
	{
		Matrix6x6d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[j][k];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __M12x6_M6x12_Multiply(const Matrix12x6d& A, const Matrix6x12d& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x9d __M9x6_M6x9_Multiply(const Matrix9x6d& A, const Matrix6x9d& B) {
		Matrix9x9d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x6d __M6x3_M3x6_Multiply(const Matrix6x3d& A, const Matrix3x6d& B) {
		Matrix6x6d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __s_M12x12_Multiply(const Matrix12x12d& A, const double& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ Matrix9x9d __s_M9x9_Multiply(const Matrix9x9d& A, const double& B)
	{
		Matrix9x9d output;
		for (int i = 0; i < 9; i++)
			for (int j = 0; j < 9; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ Matrix6x6d __s_M6x6_Multiply(const Matrix6x6d& A, const double& B)
	{
		Matrix6x6d output;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ void __Determiant(const Matrix3x3d& input, double& determinant) {
		determinant = input.m[0][0] * input.m[1][1] * input.m[2][2] +
			input.m[1][0] * input.m[2][1] * input.m[0][2] +
			input.m[2][0] * input.m[0][1] * input.m[1][2] -
			input.m[2][0] * input.m[1][1] * input.m[0][2] -
			input.m[0][0] * input.m[1][2] * input.m[2][1] -
			input.m[0][1] * input.m[1][0] * input.m[2][2];
	}

	__device__ __host__ double __Determiant(const Matrix3x3d& input) {
		return input.m[0][0] * input.m[1][1] * input.m[2][2] +
			input.m[1][0] * input.m[2][1] * input.m[0][2] +
			input.m[2][0] * input.m[0][1] * input.m[1][2] -
			input.m[2][0] * input.m[1][1] * input.m[0][2] -
			input.m[0][0] * input.m[1][2] * input.m[2][1] -
			input.m[0][1] * input.m[1][0] * input.m[2][2];
	}

	//__device__ __host__ void __Inverse(const Matrix3x3d& input, Matrix3x3d& output) {
	//	for (int i = 0;i < 3;i++) {
	//		for (int j = 0;j < 3;j++) {
	//			output.m[i][j] = input.m[i][j];
	//		}
	//	}

	//	int swapR[3], swapC[3];
	//	int pivot[3] = { 0 };

	//	for (int i = 0; i < 3; ++i) {
	//		int pr, pc;
	//		double maxValue = -1e32;
	//		for (int j = 0; j < 3; ++j) {
	//			if (pivot[j] != 1) {
	//				for (int k = 0; k < 3; ++k) {
	//					if (pivot[k] == 0) {
	//						if (output.m[j][k] > maxValue) {
	//							maxValue = output.m[j][k];
	//							pr = j;
	//							pc = k;
	//						}
	//					}
	//				}
	//			}
	//		}
	//		if (pr != pc) {
	//			double pv;
	//			for (int j = 0; j < 3; ++j) {
	//				pv = output.m[pr][j];
	//				output.m[pr][j] = output.m[pc][j];
	//				output.m[pc][j] = pv;
	//			}
	//		}
	//		swapC[i] = pc;
	//		swapR[i] = pr;
	//		++pivot[i];
	//		double inv = 1.f / output.m[pc][pc];
	//		output.m[pc][pc] = 1;
	//		for (int j = 0; j < 3; ++j) {
	//			output.m[pc][j] *= inv;
	//		}
	//		for (int j = 0; j < 3; ++j) {
	//			if (j != pc) {
	//				double powerRatio = output.m[j][pc];
	//				output.m[j][pc] = 0.f;
	//				for (int k = 0; k < 3; ++k) {
	//					output.m[j][k] -= output.m[pc][k] * powerRatio;
	//				}
	//			}
	//		}
	//	}
	//	for (int i = 0; i < 3; ++i) {
	//		if (swapR[i] != swapC[i]) {
	//			double pv;
	//			for (int j = 0; j < 3; ++j) {
	//				pv = output.m[j][swapC[i]];
	//				output.m[j][swapC[i]] = output.m[j][swapR[i]];
	//				output.m[j][swapR[i]] = pv;
	//			}
	//		}
	//	}
	//}

	__device__ __host__ void __Inverse(const Matrix3x3d& input, Matrix3x3d& result) {
		double eps = 1e-15;
		const int dim = 3;
		double mat[dim * 2][dim * 2];
		for (int i = 0;i < dim; i++)
		{
			for (int j = 0;j < 2 * dim; j++)
			{
				if (j < dim)
				{
					mat[i][j] = input.m[i][j];//[i, j];
				}
				else
				{
					mat[i][j] = j - dim == i ? 1 : 0;
				}
			}
		}

		for (int i = 0;i < dim; i++)
		{
			if (abs(mat[i][i]) < eps)
			{
				int j;
				for (j = i + 1; j < dim; j++)
				{
					if (abs(mat[j][i]) > eps) break;
				}
				if (j == dim) return;
				for (int r = i; r < 2 * dim; r++)
				{
					mat[i][r] += mat[j][r];
				}
			}
			double ep = mat[i][i];
			for (int r = i; r < 2 * dim; r++)
			{
				mat[i][r] /= ep;
			}

			for (int j = i + 1; j < dim; j++)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}

		for (int i = dim - 1; i >= 0; i--)
		{
			for (int j = i - 1; j >= 0; j--)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}


		for (int i = 0;i < dim; i++)
		{
			for (int r = dim; r < 2 * dim; r++)
			{
				result.m[i][r - dim] = mat[i][r];
			}
		}
	}

	__device__ __host__ void __Inverse2x2(const Matrix2x2d& input, Matrix2x2d& result) {
		double eps = 1e-15;
		const int dim = 2;
		double mat[dim * 2][dim * 2];
		for (int i = 0;i < dim; i++)
		{
			for (int j = 0;j < 2 * dim; j++)
			{
				if (j < dim)
				{
					mat[i][j] = input.m[i][j];//[i, j];
				}
				else
				{
					mat[i][j] = j - dim == i ? 1 : 0;
				}
			}
		}

		for (int i = 0;i < dim; i++)
		{
			if (abs(mat[i][i]) < eps)
			{
				int j;
				for (j = i + 1; j < dim; j++)
				{
					if (abs(mat[j][i]) > eps) break;
				}
				if (j == dim) return;
				for (int r = i; r < 2 * dim; r++)
				{
					mat[i][r] += mat[j][r];
				}
			}
			double ep = mat[i][i];
			for (int r = i; r < 2 * dim; r++)
			{
				mat[i][r] /= ep;
			}

			for (int j = i + 1; j < dim; j++)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}

		for (int i = dim - 1; i >= 0; i--)
		{
			for (int j = i - 1; j >= 0; j--)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}


		for (int i = 0;i < dim; i++)
		{
			for (int r = dim; r < 2 * dim; r++)
			{
				result.m[i][r - dim] = mat[i][r];
			}
		}
	}

	__device__ __host__ double __f(const double& x, const double& a, const double& b, const double& c, const double& d) {
		double f = a * x * x * x + b * x * x + c * x + d;
		return f;
	}

	__device__ __host__ double __df(const double& x, const double& a, const double& b, const double& c) {
		double df = 3 * a * x * x + 2 * b * x + c;
		return df;
	}

	__device__ __host__ void __NewtonSolverForCubicEquation(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS)
	{
		//double EPS = 1e-6;
		double DX = 0;
		//double results[3];
		num_solutions = 0;
		double specialPoint = -b / a / 3;
		double pos[2];
		int solves = 1;
		double delta = 4 * b * b - 12 * a * c;
		if (delta > 0) {
			pos[0] = (sqrt(delta) - 2 * b) / 6 / a;
			pos[1] = (-sqrt(delta) - 2 * b) / 6 / a;
			double v1 = __f(pos[0], a, b, c, d);
			double v2 = __f(pos[1], a, b, c, d);
			if (__mabs(v1) < EPS * EPS) {
				v1 = 0;
			}
			if (__mabs(v2) < EPS * EPS) {
				v2 = 0;
			}
			double sign = v1 * v2;
			DX = (pos[0] - pos[1]);
			if (sign <= 0) {
				solves = 3;
			}
			else if (sign > 0) {
				if ((a < 0 && __f(pos[0], a, b, c, d) > 0) || (a > 0 && __f(pos[0], a, b, c, d) < 0)) {
					DX = -DX;
				}
			}
		}
		else if (delta == 0) {
			if (__mabs(__f(specialPoint, a, b, c, d)) < EPS * EPS) {
				for (int i = 0; i < 3; i++) {
					double tempReuslt = specialPoint;
					results[num_solutions] = tempReuslt;
					num_solutions++;
				}
				return;
			}
			if (a > 0) {
				if (__f(specialPoint, a, b, c, d) > 0) {
					DX = 1;
				}
				else if (__f(specialPoint, a, b, c, d) < 0) {
					DX = -1;
				}
			}
			else if (a < 0) {
				if (__f(specialPoint, a, b, c, d) > 0) {
					DX = -1;
				}
				else if (__f(specialPoint, a, b, c, d) < 0) {
					DX = 1;
				}
			}

		}

		double start = specialPoint - DX;
		double x0 = start;
		double result[3];

		for (int i = 0; i < solves; i++) {
			double x1 = 0;
			int itCount = 0;
			do
			{
				if (itCount)
					x0 = x1;

				x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
				itCount++;

			} while (__mabs(x1 - x0) > EPS && itCount < 100000);
			results[num_solutions] = (x1);
			num_solutions++;
			start = start + DX;
			x0 = start;
		}
	}

	__device__ __host__ void __SolverForCubicEquation(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS) {
		double A = b * b - 3 * a * c;
		double B = b * c - 9 * a * d;
		double C = c * c - 3 * b * d;
		double delta = B * B - 4 * A * C;
		num_solutions = 0;
		if (abs(A) < EPS * EPS && abs(B) < EPS * EPS) {
			results[0] = -b / 3.0 / a;
			results[1] = results[0];
			results[2] = results[0];
			num_solutions = 3;
		}
		else if (abs(delta) <= EPS * EPS) {
			double K = B / A;
			results[0] = -b / a + K;
			results[1] = -K / 2.0;
			results[2] = results[1];
			num_solutions = 3;
		}
		else if (delta < -EPS * EPS) {
			double T = (2 * A * b - 3 * a * B) / (2 * A * sqrt(A));
			double theta = acos(T);
			results[0] = (-b - 2 * sqrt(A) * cos(theta / 3.0)) / (3 * a);
			results[1] = (-b + sqrt(A) * (cos(theta / 3.0) + sqrt(3.0) * sin(theta / 3.0))) / (3 * a);
			results[2] = (-b + sqrt(A) * (cos(theta / 3.0) - sqrt(3.0) * sin(theta / 3.0))) / (3 * a);
			num_solutions = 3;
		}
		else if (delta > EPS * EPS) {
			double Y1 = A * b + 3 * a * (-B + sqrt(delta)) / 2;
			double Y2 = A * b + 3 * a * (-B - sqrt(delta)) / 2;

			results[0] = -b - cbrt(Y1) - cbrt(Y2);
			num_solutions = 1;
		}
	}

	__device__ __host__ Vector9 __Mat3x3_to_vec9_double(const Matrix3x3d& F) {

		Vector9 result;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				result.v[i * 3 + j] = F.m[j][i];
			}
		}
		return result;
	}

	__device__ __host__ void __normalized_vec9_double(Vector9& v9) {

		double length = 0;
		for (int i = 0;i < 9;i++) {
			length += v9.v[i] * v9.v[i];
		}
		length = 1.0 / sqrt(length);
		for (int i = 0;i < 9;i++) {
			v9.v[i] = v9.v[i] * length;
		}
	}

	__device__ __host__ Vector6 __Mat3x2_to_vec6_double(const Matrix3x2d& F) {

		Vector6 result;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 3; j++) {
				result.v[i * 3 + j] = F.m[j][i];
			}
		}
		return result;
	}

	__device__ __host__ Matrix3x3d __vec9_to_Mat3x3_double(const double vec9[9]) {
		Matrix3x3d mat;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				mat.m[j][i] = vec9[i * 3 + j];
			}
		}
		return mat;
	}

	__device__ __host__ Matrix2x2d __vec4_to_Mat2x2_double(const double vec4[4]) {
		Matrix2x2d mat;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				mat.m[j][i] = vec4[i * 2 + j];
			}
		}
		return mat;
	}

	__device__ void SVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma) {
		float U[9], V[9], S[3];
		svd(F.m[0][0], F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0], F.m[2][1], F.m[2][2],
			U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8],
			S[0], S[1], S[2],
			V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

		Uout.m[0][0] = U[0];Uout.m[0][1] = U[3];Uout.m[0][2] = U[6];
		Uout.m[1][0] = U[1];Uout.m[1][1] = U[4];Uout.m[1][2] = U[7];
		Uout.m[2][0] = U[2];Uout.m[2][1] = U[5];Uout.m[2][2] = U[8];

		Vout.m[0][0] = V[0];Vout.m[0][1] = V[3];Vout.m[0][2] = V[6];
		Vout.m[1][0] = V[1];Vout.m[1][1] = V[4];Vout.m[1][2] = V[7];
		Vout.m[2][0] = V[2];Vout.m[2][1] = V[5];Vout.m[2][2] = V[8];

		Sigma.m[0][0] = S[0];Sigma.m[0][1] = 0;Sigma.m[0][2] = 0;
		Sigma.m[1][0] = 0;Sigma.m[1][1] = S[1];Sigma.m[1][2] = 0;
		Sigma.m[2][0] = 0;Sigma.m[2][1] = 0;Sigma.m[2][2] = S[2];

		double du, dv;
		__Determiant(Uout, du);
		__Determiant(Vout, dv);

		if (du < 0) {
			Uout.m[0][2] *= -1;
			Uout.m[1][2] *= -1;
			Uout.m[2][2] *= -1;
			Sigma.m[2][2] *= -1;
		}
		if (dv < 0) {
			Vout.m[0][2] *= -1;
			Vout.m[1][2] *= -1;
			Vout.m[2][2] *= -1;
			Sigma.m[2][2] *= -1;
		}
	}

	__device__ __host__ void __makePD2x2(const double& a00, const double& a01, const double& a10, const double& a11, double eigenValues[2], int& num, double2 eigenVectors[2], double eps) {
		double b = -(a00 + a11), c = a00 * a11 - a10 * a01;
		double existEv = b * b - 4 * c;
		if (abs(a01) < eps || abs(a10) < eps) {
			if (a00 > 0) {
				eigenValues[num] = a00;
				eigenVectors[num].x = 1;
				eigenVectors[num].y = 0;
				num++;
			}
			if (a11 > 0) {
				eigenValues[num] = a11;
				eigenVectors[num].x = 0;
				eigenVectors[num].y = 1;
				num++;
			}
		}
		else {
			if (existEv > 0) {
				num = 2;
				eigenValues[0] = (-b - sqrt(existEv)) / 2;
				eigenVectors[0].x = 1;
				eigenVectors[0].y = (eigenValues[0] - a00) / a01;
				double length = sqrt(eigenVectors[0].x * eigenVectors[0].x + eigenVectors[0].y * eigenVectors[0].y);
				eigenVectors[0].x /= length;
				eigenVectors[0].y /= length;

				eigenValues[1] = (-b + sqrt(existEv)) / 2;
				eigenVectors[1].x = 1;
				eigenVectors[1].y = (eigenValues[1] - a00) / a01;
				length = sqrt(eigenVectors[1].x * eigenVectors[1].x + eigenVectors[1].y * eigenVectors[1].y);
				eigenVectors[1].x /= length;
				eigenVectors[1].y /= length;
			}
			else if (existEv == 0) {
				num = 1;
				eigenValues[0] = (-b - sqrt(existEv)) / 2;
				eigenVectors[0].x = 1;
				eigenVectors[0].y = (eigenValues[0] - a00) / a01;
				double length = sqrt(eigenVectors[0].x * eigenVectors[0].x + eigenVectors[0].y * eigenVectors[0].y);
				eigenVectors[0].x /= length;
				eigenVectors[0].y /= length;
			}
			else {
				num = 0;
			}
		}
	}
}
