#pragma once
#ifndef  __FEM_ENERGY_CUH__
#define __FEM_ENERGY_CUH__
#include <cstdint>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "gpu_eigen_libs.cuh"

__device__ __host__
void __calculateDms3D_double(const double3* vertexes, const uint4& index, __GEIGEN__::Matrix3x3d& M);

__device__ 
__GEIGEN__::Matrix9x9d __project_StabbleNHK_H_3D(const __GEIGEN__::Matrix3x3d& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate, const double& volumRate);

__device__
__GEIGEN__::Matrix9x9d project_ARAP_H_3D(const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate);

__device__
__GEIGEN__::Matrix3x3d computePEPF_ARAP_double(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double& lengthRate);

__device__ 
__GEIGEN__::Matrix9x9d __project_ANIOSI5_H_3D(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, const double3& fiber_direction, const double& scale, const double& contract_length);

__device__ 
__GEIGEN__::Matrix3x3d __computePEPF_StableNHK3D_double(const __GEIGEN__::Matrix3x3d& F, const __GEIGEN__::Matrix3x3d& Sigma, const __GEIGEN__::Matrix3x3d& U, const __GEIGEN__::Matrix3x3d& V, double lengthRate, double volumRate);

__device__ 
__GEIGEN__::Matrix3x3d __computePEPF_Aniostropic3D_double(const __GEIGEN__::Matrix3x3d& F, double3 fiber_direction, const double& scale, const double& contract_length);

__device__ 
double __cal_StabbleNHK_energy_3D(const double3* vertexes, const uint4& tetrahedra, const __GEIGEN__::Matrix3x3d& DmInverse, const double& volume, const double& lenRate, const double& volRate);

__device__
double __cal_ARAP_energy_3D(const double3* vertexes, const uint4& tetrahedra, const __GEIGEN__::Matrix3x3d& DmInverse, const double& volume, const double& lenRate);

__device__
__GEIGEN__::Matrix9x12d __computePFDsPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm);

__device__
__GEIGEN__::Matrix6x12d __computePFDsPX3D_6x12_double(const __GEIGEN__::Matrix2x2d& InverseDm);

__device__
__GEIGEN__::Matrix6x9d __computePFDsPX3D_6x9_double(const __GEIGEN__::Matrix2x2d& InverseDm);

__device__
__GEIGEN__::Matrix3x6d __computePFDsPX3D_3x6_double(const double& InverseDm);

__device__
__GEIGEN__::Matrix9x12d __computePFDmPX3D_double(const __GEIGEN__::Matrix12x9d& PDmPx, const __GEIGEN__::Matrix3x3d& Ds, const __GEIGEN__::Matrix3x3d& DmInv);

__device__
__GEIGEN__::Matrix6x12d __computePFDmPX3D_6x12_double(const __GEIGEN__::Matrix12x4d& PDmPx, const __GEIGEN__::Matrix3x2d& Ds, const __GEIGEN__::Matrix2x2d& DmInv);

__device__
__GEIGEN__::Matrix3x6d __computePFDmPX3D_3x6_double(const __GEIGEN__::Vector6& PDmPx, const double3& Ds, const double& DmInv);

__device__
__GEIGEN__::Matrix6x9d __computePFDmPX3D_6x9_double(const __GEIGEN__::Matrix9x4d& PDmPx, const __GEIGEN__::Matrix3x2d& Ds, const __GEIGEN__::Matrix2x2d& DmInv);

__device__
__GEIGEN__::Matrix9x12d __computePFPX3D_double(const __GEIGEN__::Matrix3x3d& InverseDm);

__global__
void _calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    __GEIGEN__::Matrix12x12d* Hessians, uint32_t offset, const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double IPC_dt);

__global__
void _calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate);

double calculateVolum(const double3* vertexes, const uint4& index);
#endif // ! __FEM_ENERGY_CUH__
