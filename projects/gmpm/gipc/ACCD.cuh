#pragma once
#include <cuda_runtime.h>

__device__
double point_triangle_ccd(
    const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double eta, double thickness);

__device__
double edge_edge_ccd(
    const double3& _ea0,
    const double3& _ea1,
    const double3& _eb0,
    const double3& _eb1,
    const double3& _dea0,
    const double3& _dea1,
    const double3& _deb0,
    const double3& _deb1,
    double eta, double thickness);

__device__ 
double doCCDVF(const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double errorRate, double thickness);