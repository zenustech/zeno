#include "mlbvh.cuh"
#include <cmath>
#include "cuda_tools.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include<iostream>
#include<fstream>
#include "gpu_eigen_libs.cuh"
#include "zensim/container/Bvh.hpp"
#include "zensim/geometry/BoundingVolumeInterface.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

template <class F>
__device__ __host__
inline F __m_min(F a, F b) {
    return a > b ? b : a;
}


template <class F>
__device__ __host__
inline F __m_max(F a, F b) {
    return a > b ? a : b;
}

__device__ __host__
inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept
{
    AABB merged;
    merged.upper.x = __m_max(lhs.upper.x, rhs.upper.x);
    merged.upper.y = __m_max(lhs.upper.y, rhs.upper.y);
    merged.upper.z = __m_max(lhs.upper.z, rhs.upper.z);
    merged.lower.x = __m_min(lhs.lower.x, rhs.lower.x);
    merged.lower.y = __m_min(lhs.lower.y, rhs.lower.y);
    merged.lower.z = __m_min(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__
inline bool overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

__device__ __host__
inline double3 centroid(const AABB& box) noexcept
{
    double3 c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

__device__ __host__
inline std::uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__
inline std::uint32_t morton_code(double x, double y, double z, double resolution = 1024.0) noexcept
{
    x = __m_min(__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __m_min(__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __m_min(__m_max(z * resolution, 0.0), resolution - 1.0);
    
    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(z));
    
    std::uint32_t mchash = ((xx << 2) + (yy << 1) + zz);

    return mchash;
}

__device__ __host__
void AABB::combines(const double& x, const double& y, const double& z)
{
    lower = make_double3(__m_min(lower.x, x), __m_min(lower.y, y), __m_min(lower.z, z));
    upper = make_double3(__m_max(upper.x, x), __m_max(upper.y, y), __m_max(upper.z, z));
}

__device__ __host__
void AABB::combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz)
{
    lower = make_double3(__m_min(lower.x, x), __m_min(lower.y, y), __m_min(lower.z, z));
    upper = make_double3(__m_max(upper.x, xx), __m_max(upper.y, yy), __m_max(upper.z, zz));
}

__host__ __device__  
void AABB::combines(const AABB& aabb) {
    lower = make_double3(__m_min(lower.x, aabb.lower.x), __m_min(lower.y, aabb.lower.y), __m_min(lower.z, aabb.lower.z));
    upper = make_double3(__m_max(upper.x, aabb.upper.x), __m_max(upper.y, aabb.upper.y), __m_max(upper.z, aabb.upper.z));
}

__host__ __device__ 
double3 AABB::center() {
    return make_double3((upper.x + lower.x) * 0.5, (upper.y + lower.y) * 0.5, (upper.z + lower.z) * 0.5);
}

__device__ __host__
AABB::AABB()
{
    lower = make_double3(1e32, 1e32, 1e32);
    upper = make_double3(-1e32, -1e32, -1e32);
}

__device__
inline int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept
{
    return ::__clz(lhs ^ rhs);
}
__device__
inline int common_upper_bits(const uint64_t lhs, const unsigned long long int rhs) noexcept
{
    return ::__clzll((unsigned long long int)lhs ^ (unsigned long long int)rhs);
}


__device__
inline uint2 determine_range(const uint64_t* node_code,
    const unsigned int num_leaves, unsigned int idx)
{
    if (idx == 0)
    {
        return make_uint2(0, num_leaves - 1);
    }

    // determine direction of the range
    const uint64_t self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = __m_min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while (delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while (t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if (delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0)
    {
        unsigned int temp_jdx = jdx;
        jdx = idx;
        idx = temp_jdx;
    }
    return make_uint2(idx, jdx);
}

__device__
inline unsigned int find_split(const uint64_t* node_code, const unsigned int num_leaves,
    const unsigned int first, const unsigned int last) noexcept
{
    const uint64_t first_code = node_code[first];
    const uint64_t last_code = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

__device__
void _d_PP(const double3& v0, const double3& v1, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1));
}

__device__
void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1), __GEIGEN__::__minus(v3, v1));
    double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__
void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0))) / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v1));
}

__device__
void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}


__device__
void _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__
double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1)) * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

__device__
double _compute_epx_cp(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1)) * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

__device__
int _dType_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
    __GEIGEN__::Matrix3x3d D, D1, D2;

    __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
    __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
    __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

    double2 param[3];
    param[0].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
    param[0].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

    if (param[0].x > 0 && param[0].x < 1 && param[0].y >= 0) {
        return 3; // PE v1v2
    }
    else {
        basis0 = __GEIGEN__::__minus(v3, v2);
        basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
        basis2 = __GEIGEN__::__minus(v0, v2);

        __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
        __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
        __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

        param[1].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
        param[1].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

        if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
            return 4; // PE v2v3
        }
        else {
            basis0 = __GEIGEN__::__minus(v1, v3);
            basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
            basis2 = __GEIGEN__::__minus(v0, v3);

            __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
            __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
            __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

            param[2].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
            param[2].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

            if (param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0) {
                return 5; // PE v3v1
            }
            else {
                if (param[0].x <= 0.0 && param[2].x >= 1.0) {
                    return 0; // PP v1
                }
                else if (param[1].x <= 0.0 && param[0].x >= 1.0) {
                    return 1; // PP v2
                }
                else if (param[2].x <= 0.0 && param[1].x >= 1.0) {
                    return 2; // PP v3
                }
                else {
                    return 6; // PT
                }
            }
        }
    }
}

__device__
int _dType_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 u = __GEIGEN__::__minus(v1, v0);
    double3 v = __GEIGEN__::__minus(v3, v2);
    double3 w = __GEIGEN__::__minus(v0, v2);

    double a = __GEIGEN__::__squaredNorm(u);
    double b = __GEIGEN__::__v_vec_dot(u, v);
    double c = __GEIGEN__::__squaredNorm(v);
    double d = __GEIGEN__::__v_vec_dot(u, w);
    double e = __GEIGEN__::__v_vec_dot(v, w);

    double D = a * c - b * b; // always >= 0
    double tD = D; // tc = tN / tD, default tD = D >= 0
    double sN, tN;
    int defaultCase = 8;
    sN = (b * e - c * d);
    if (sN <= 0.0) { // sc < 0 => the s=0 edge is visible
        tN = e;
        tD = c;
        defaultCase = 2;
    }
    else if (sN >= D) { // sc > 1  => the s=1 edge is visible
        tN = e + b;
        tD = c;
        defaultCase = 5;
    }
    else {
        tN = (a * e - b * d);
        if (tN > 0.0 && tN < tD && (__GEIGEN__::__v_vec_dot(w, __GEIGEN__::__v_vec_cross(u, v)) == 0.0 || __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(u, v)) < 1.0e-20 * a * c)) {
            if (sN < D / 2) {
                tN = e;
                tD = c;
                defaultCase = 2;
            }
            else {
                tN = e + b;
                tD = c;
                defaultCase = 5;
            }
        }
    }

    if (tN <= 0.0) { 
        if (-d <= 0.0) {
            return 0;
        }
        else if (-d >= a) {
            return 3;
        }
        else {
            return 6;
        }
    }
    else if (tN >= tD) { 
        if ((-d + b) <= 0.0) {
            return 1;
        }
        else if ((-d + b) >= a) {
            return 4;
        }
        else {
            return 7;
        }
    }

    return defaultCase;
}


__device__ 
inline bool _checkPTintersection(const double3* _vertexes, const uint32_t& id0, const uint32_t& id1, const uint32_t& id2, const uint32_t& id3, const double& dHat, uint32_t* _cpNum, int* _mInx, int4* _collisionPair, int4* _ccd_collisionPair) noexcept
{
    using namespace zs;
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];
    auto tovec = [](const double3& v) {
        return zs::vec<double, 3>{v.x, v.y, v.z};
    };
    auto v0_ = tovec(v0);
    auto v1_ = tovec(v1);
    auto v2_ = tovec(v2);
    auto v3_ = tovec(v3);

    // int dtype = _dType_PT(v0, v1, v2, v3);
    int dtype = pt_distance_type(v0_, v1_, v2_, v3_);

    double d{};
    switch (dtype) {
    case 0: {
        // _d_PP(v0, v1, d);
        d = dist2_pp(v0_, v1_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);         
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, -1, -1);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
        }
        break;
    }

    case 1: {
        // _d_PP(v0, v2, d);
        d = dist2_pp(v0_, v2_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);         
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, -1);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
        }
        break;
    }

    case 2: {
        // _d_PP(v0, v3, d);
        d = dist2_pp(v0_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, -1);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
        }
        break;
    }

    case 3: {
        // _d_PE(v0, v1, v2, d);
        d = dist2_pe(v0_, v1_, v2_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, -1);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
        }
        break;
    }

    case 4: {
        // _d_PE(v0, v2, v3, d);
        d = dist2_pe(v0_, v2_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, -1);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
        }
        break;
    }

    case 5: {
        // _d_PE(v0, v3, v1, d);
        d = dist2_pe(v0_, v3_, v1_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, id1, -1);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
        }
        break;
    }

    case 6: {
        // _d_PT(v0, v1, v2, v3, d);
        d = dist2_pt(v0_, v1_, v2_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
            _mInx[cdp_idx] = atomicAdd(_cpNum + 4, 1);
        }
        break;
    }

    default:
        break;
    }
}

__device__
inline bool _checkPTintersection_fullCCD(const double3* _vertexes, const uint32_t& id0, const uint32_t& id1, const uint32_t& id2, const uint32_t& id3, const double& dHat, uint32_t* _cpNum, int4* _ccd_collisionPair) noexcept
{
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];

    int dtype = _dType_PT(v0, v1, v2, v3);

    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    double sign = __GEIGEN__::__v_vec_dot(nVec, basis2);

    if (dtype==6&&(sign <0)) {
        return;
    }

    _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-id0 - 1, id1, id2, id3);
}

__device__
inline bool _checkEEintersection(const double3* _vertexes, const double3* _rest_vertexes, const uint32_t& id0, const uint32_t& id1, const uint32_t& id2, const uint32_t& id3, const double& dHat, uint32_t* _cpNum, int* MatIndex, int4* _collisionPair, int4* _ccd_collisionPair, int edgeNum) noexcept
{
    using namespace zs;
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];
    auto tovec = [](const double3& v) {
        return zs::vec<double, 3>{v.x, v.y, v.z};
    };
    auto v0_ = tovec(v0);
    auto v1_ = tovec(v1);
    auto v2_ = tovec(v2);
    auto v3_ = tovec(v3);
    auto rv0_ = tovec(_rest_vertexes[id0]);
    auto rv1_ = tovec(_rest_vertexes[id1]);
    auto rv2_ = tovec(_rest_vertexes[id2]);
    auto rv3_ = tovec(_rest_vertexes[id3]);

    double eeSqureNCross = cn2_ee(v0_, v1_, v2_, v3_);
    double eps_x = mollifier_threshold_ee(rv0_, rv1_, rv2_, rv3_);
    int add_e = -1;
    bool mollify = eeSqureNCross < eps_x;

    // int dtype = _dType_EE(v0, v1, v2, v3);
    int dtype = ee_distance_type(v0_, v1_, v2_, v3_);
    double d = 100.0;
    switch (dtype) {
    case 0: {
        // _d_PP(v0, v2, d);
        d = dist2_pp(v0_, v2_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, -id2 - 1, -id1 - 1, -id3 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
        }
        break;
    }
    
    case 1: {
        d = dist2_pp(v0_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, -id3 - 1, -id1 - 1, -id2 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
        }
        break;
    }
        
    case 2: {
        d = dist2_pe(v0_, v2_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, -id2 - 1, id3, -id1 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
        }
        break;
    }

    case 3: {
        d = dist2_pp(v1_, v2_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id1 - 1, -id2 - 1, -id0 - 1, -id3 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
        }
        break;
    }

    case 4: {
        // _d_PP(v1, v3, d);
        d = dist2_pp(v1_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id1 - 1, -id3 - 1, -id0 - 1, -id2 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
        }
        break;
    }

    case 5: {
        // _d_PE(v1, v2, v3, d);
        d = dist2_pe(v1_, v2_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id1 - 1, -id2 - 1, id3, -id0 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
        }
        break;
    }

    case 6: {
        // _d_PE(v2, v0, v1, d);
        d = dist2_pe(v2_, v0_, v1_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id2 - 1, -id0 - 1, id1, -id3 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
        }
        break;
    }

    case 7: {
        // _d_PE(v3, v0, v1, d);
        d = dist2_pe(v3_, v0_, v1_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                _collisionPair[cdp_idx] = make_int4(-id3 - 1, -id0 - 1, id1, -id2 - 1);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
        }
        break;
    }

    case 8: {
        d = dist2_ee(v0_, v1_, v2_, v3_);
        if (d < dHat) {
            int cdp_idx = atomicAdd(_cpNum, 1);
            _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
            if (mollify) {
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                _collisionPair[cdp_idx] = make_int4(id0, id1, id2, -id3 - 1);
            }
            else {
                _collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
        }
        break;
    }

    default:
        break;
    }
}

__global__
void _reduct_max_box(AABB* _leafBoxes, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ AABB tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    AABB temp = _leafBoxes[idx];

    __threadfence();

    double xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
    double xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
    //printf("%f   %f    %f   %f   %f    %f\n", xmin, ymin, zmin, xmax, ymax, zmax);
    //printf("%f   %f    %f\n", xmax, ymax, zmax);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((number - idof + 31) >> 5);
        if (warpId == warpNum - 1) {
            tidNum = number - idof - (warpNum - 1) * 32;
        }
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < tidNum; i = (i << 1)) {
        temp.combines(__shfl_down(xmin, i), __shfl_down(ymin, i), __shfl_down(zmin, i),
            __shfl_down(xmax, i), __shfl_down(ymax, i), __shfl_down(zmax, i));
        if (warpTid + i < tidNum) {
            xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
            xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        }
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
        xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp.combines(__shfl_down(xmin, i), __shfl_down(ymin, i), __shfl_down(zmin, i),
                __shfl_down(xmax, i), __shfl_down(ymax, i), __shfl_down(zmax, i));
            if (threadIdx.x + i < warpNum) {
                xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
                xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
            }
        }
    }
    if (threadIdx.x == 0) {
        _leafBoxes[blockIdx.x] = temp;
    }
}

template <class element_type>
__global__
void _calcLeafBvs(const double3* _vertexes, const element_type* _elements, AABB* _bvs, int faceNum, int type = 0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= faceNum) return;
    AABB _bv;

    element_type _e = _elements[idx];
    double3 _v = _vertexes[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    if (type == 0) {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
    }
    _bvs[idx] = _bv;
}

template <class element_type>
__global__
void _calcLeafBvs_ccd(const double3* _vertexes, const double3* _moveDir, double alpha, const element_type* _elements, AABB* _bvs, int faceNum, int type = 0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= faceNum) return;
    AABB _bv;

    element_type _e = _elements[idx];
    double3 _v = _vertexes[_e.x];
    double3 _mvD = _moveDir[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);


    _v = _vertexes[_e.y];
    _mvD = _moveDir[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    if (type == 0) {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _mvD = _moveDir[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
        _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    }
    _bvs[idx] = _bv;
}

__global__
void _calcMChash(uint64_t* _MChash, AABB* _bvs, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    AABB maxBv = _bvs[0];
    double3 SceneSize = make_double3(maxBv.upper.x - maxBv.lower.x, maxBv.upper.y - maxBv.lower.y, maxBv.upper.z - maxBv.lower.z);
    double3 centerP = _bvs[idx + number - 1].center();
    double3 offset = make_double3(centerP.x - maxBv.lower.x, centerP.y - maxBv.lower.y, centerP.z - maxBv.lower.z);
    
    //printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = morton_code(offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    _MChash[idx] = mc64;
}

__global__
void _calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    if (idx < number - 1) {
        _nodes[idx].left_idx = 0xFFFFFFFF;
        _nodes[idx].right_idx = 0xFFFFFFFF;
        _nodes[idx].parent_idx = 0xFFFFFFFF;
        _nodes[idx].element_idx = 0xFFFFFFFF;
    }
    int l_idx = idx + number - 1;
    _nodes[l_idx].left_idx = 0xFFFFFFFF;
    _nodes[l_idx].right_idx = 0xFFFFFFFF;
    _nodes[l_idx].parent_idx = 0xFFFFFFFF;
    _nodes[l_idx].element_idx = _indices[idx];
}




__global__
void _calcInternalNodes(Node* _nodes, const uint64_t* _MChash, int number) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number - 1) return;
    const uint2 ij = determine_range(_MChash, number, idx);
    const unsigned int gamma = find_split(_MChash, number, ij.x, ij.y);

    _nodes[idx].left_idx = gamma;
    _nodes[idx].right_idx = gamma + 1;
    if (__m_min(ij.x, ij.y) == gamma)
    {
        _nodes[idx].left_idx += number - 1;
    }
    if (__m_max(ij.x, ij.y) == gamma + 1)
    {
        _nodes[idx].right_idx += number - 1;
    }
    _nodes[_nodes[idx].left_idx].parent_idx = idx;
    _nodes[_nodes[idx].right_idx].parent_idx = idx;
}

__global__
void _calcInternalAABB(const Node* _nodes, AABB* _bvs, uint32_t* flags, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    idx = idx + number - 1;

    uint32_t parent = _nodes[idx].parent_idx;
    while (parent != 0xFFFFFFFF) // means idx == 0
    {
        const int old = atomicCAS(flags + parent, 0xFFFFFFFF, 0);
        if (old == 0xFFFFFFFF)
        {
            return;
        }

        const uint32_t lidx = _nodes[parent].left_idx;
        const uint32_t ridx = _nodes[parent].right_idx;

        const AABB lbox = _bvs[lidx];
        const AABB rbox = _bvs[ridx];
        _bvs[parent] = merge(lbox, rbox);

        __threadfence();

        parent = _nodes[parent].parent_idx;

    }
}

__global__
void _sortBvs(const uint32_t* _indices, AABB* _bvs, AABB* _temp_bvs, int number) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    _bvs[idx] = _temp_bvs[_indices[idx]];
}

__global__
void _selfQuery_vf(const double3* _vertexes, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes, int4* _collisionPair, int4* _ccd_collisionPair, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;
      
    AABB _bv;
    idx = _surfVerts[idx];
    _bv.upper = _vertexes[idx];
    _bv.lower = _vertexes[idx];
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);//0.001 * sqrt(bboxDiagSize2);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                    _checkPTintersection(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair);
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                    _checkPTintersection(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair);
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
} 

__global__
void _selfQuery_vf_ccd(const double3* _vertexes, const double3* moveDir, double alpha, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes, int4* _ccd_collisionPair, uint32_t* _cpNum, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    AABB _bv;
    idx = _surfVerts[idx];
    double3 current_vertex = _vertexes[idx];
    double3 mvD = moveDir[idx];
    _bv.upper = current_vertex;
    _bv.lower = current_vertex;
    _bv.combines(current_vertex.x - mvD.x * alpha, current_vertex.y - mvD.y * alpha, current_vertex.z - mvD.z * alpha);
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);//0.001 * sqrt(bboxDiagSize2);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                    _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-idx - 1, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z);
                    //_checkPTintersection_fullCCD(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, _ccd_collisionPair);
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                    _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-idx - 1, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z);
                    //_checkPTintersection_fullCCD(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, _ccd_collisionPair);
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}


__global__
void _selfQuery_ee(const double3* _vertexes, const double3* _rest_vertexes, const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _collisionPair, int4* _ccd_collisionPair, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    idx = idx + number - 1;
    AABB _bv = _bvs[idx];
    uint32_t self_eid = _nodes[idx].element_idx;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);//0.001 * sqrt(bboxDiagSize2);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;
        
        if (overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (self_eid != obj_idx) {
                    if (!(_edges[self_eid].x == _edges[obj_idx].x || _edges[self_eid].x == _edges[obj_idx].y || _edges[self_eid].y == _edges[obj_idx].x || _edges[self_eid].y == _edges[obj_idx].y || obj_idx < self_eid)) {
                        //printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
                        _checkEEintersection(_vertexes, _rest_vertexes, _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair, number);
                    }
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (self_eid != obj_idx) {
                    if (!(_edges[self_eid].x == _edges[obj_idx].x || _edges[self_eid].x == _edges[obj_idx].y || _edges[self_eid].y == _edges[obj_idx].x || _edges[self_eid].y == _edges[obj_idx].y || obj_idx < self_eid)) {
                        //printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
                        _checkEEintersection(_vertexes, _rest_vertexes, _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair, number);
                    }
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

__global__
void _selfQuery_ee_ccd(const double3* _vertexes, const double3* moveDir, double alpha, const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _ccd_collisionPair, uint32_t* _cpNum, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;
    idx = idx + number - 1;
    AABB _bv = _bvs[idx];
    uint32_t self_eid = _nodes[idx].element_idx;
    uint2 current_edge = _edges[self_eid];
    //double3 edge_tvert0 = __GEIGEN__::__minus(_vertexes[current_edge.x], __GEIGEN__::__s_vec_multiply(moveDir[current_edge.x], alpha));
    //double3 edge_tvert1 = __GEIGEN__::__minus(_vertexes[current_edge.y], __GEIGEN__::__s_vec_multiply(moveDir[current_edge.y], alpha));
    //_bv.combines(edge_tvert0.x, edge_tvert0.y, edge_tvert0.z);
    //_bv.combines(edge_tvert1.x, edge_tvert1.y, edge_tvert1.z);
    double gapl = sqrt(dHat);

    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl))
        {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (self_eid != obj_idx) {
                    if (!(current_edge.x == _edges[obj_idx].x || current_edge.x == _edges[obj_idx].y || current_edge.y == _edges[obj_idx].x || current_edge.y == _edges[obj_idx].y || obj_idx < self_eid)) {
                        _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(current_edge.x, current_edge.y, _edges[obj_idx].x, _edges[obj_idx].y);
                    }
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl))
        {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (self_eid != obj_idx) {
                    if (!(current_edge.x == _edges[obj_idx].x || current_edge.x == _edges[obj_idx].y || current_edge.y == _edges[obj_idx].x || current_edge.y == _edges[obj_idx].y || obj_idx < self_eid)) {
                        _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(current_edge.x, current_edge.y, _edges[obj_idx].x, _edges[obj_idx].y);
                    }
                }
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

///////////////////////////////////////host//////////////////////////////////////////////


AABB calcMaxBV(AABB* _leafBoxes, AABB* _tempLeafBox, const int& number) {

    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(AABB) * (threadNum >> 5);

    //AABB* _tempLeafBox;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_tempLeafBox, number * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMemcpy(_tempLeafBox, _leafBoxes + number - 1, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    
    _reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        _reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    AABB h_bv;
    cudaMemcpy(&h_bv, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempLeafBox));
    return h_bv;
}

zs::Vector<bv_t> retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, 
                          int faceNum,
                          const double3 *verts,
                          const uint3 *faces, double xi = 0.) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> ret{(std::size_t)faceNum, memsrc_e::device, 0};
        pol(zs::range(faceNum), [verts, faces,
                            bvs = proxy<space>(ret),
                            xi] ZS_LAMBDA(int ei) mutable {
        auto getV = [](const double3& v) {
            return zs::vec<double, 3>{v.x, v.y, v.z};
        };
        auto inds = faces[ei];
        auto x0 = getV(verts[inds.x]);
        auto x1 = getV(verts[inds.y]);
        auto x2 = getV(verts[inds.z]);
        bv_t bv{get_bounding_box(x0, x1)};
        merge(bv, x2);
        bv._min -= xi / 2;
        bv._max += xi / 2;
        bvs[ei] = bv;
        });
        return ret;
}

zs::Vector<bv_t> retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, 
                          int faceNum,
                          const double3 *verts,
                          const double3 *dirs,
                          const uint3 *faces, double alpha, double xi = 0.) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> ret{(std::size_t)faceNum, memsrc_e::device, 0};
        pol(zs::range(faceNum), [verts, faces, dirs,
                            bvs = proxy<space>(ret),
                            xi, alpha] ZS_LAMBDA(int ei) mutable {
        auto getV = [](const double3& v) {
            return zs::vec<double, 3>{v.x, v.y, v.z};
        };
        auto inds = faces[ei];
        auto x0 = getV(verts[inds.x]);
        auto x1 = getV(verts[inds.y]);
        auto x2 = getV(verts[inds.z]);
        auto x0_ = x0 + alpha * getV(dirs[inds.x]);
        auto x1_ = x1 + alpha * getV(dirs[inds.y]);
        auto x2_ = x2 + alpha * getV(dirs[inds.z]);
        bv_t bv{get_bounding_box(x0, x1)};
        merge(bv, x2);
        merge(bv, x0_);
        merge(bv, x1_);
        merge(bv, x2_);
        bv._min -= xi / 2;
        bv._max += xi / 2;
        bvs[ei] = bv;
        });
        return ret;
}

zs::Vector<bv_t> retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, 
                          int edgeNum,
                          const double3 *verts,
                          const uint2 *edges, double xi = 0.) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> ret{(std::size_t)edgeNum, memsrc_e::device, 0};
        pol(zs::range(edgeNum), [verts, edges,
                            bvs = proxy<space>(ret),
                            xi] ZS_LAMBDA(int ei) mutable {
        auto getV = [](const double3& v) {
            return zs::vec<double, 3>{v.x, v.y, v.z};
        };
        auto inds = edges[ei];
        auto x0 = getV(verts[inds.x]);
        auto x1 = getV(verts[inds.y]);
        bv_t bv{get_bounding_box(x0, x1)};
        bv._min -= xi / 2;
        bv._max += xi / 2;
        bvs[ei] = bv;
        });
        return ret;
}

zs::Vector<bv_t> retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, 
                          int edgeNum,
                          const double3 *verts,
                          const double3 *dirs,
                          const uint2 *edges, double alpha, double xi = 0.) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<bv_t> ret{(std::size_t)edgeNum, memsrc_e::device, 0};
        pol(zs::range(edgeNum), [verts, edges, dirs,
                            bvs = proxy<space>(ret),
                            xi, alpha] ZS_LAMBDA(int ei) mutable {
        auto getV = [](const double3& v) {
            return zs::vec<double, 3>{v.x, v.y, v.z};
        };
        auto inds = edges[ei];
        auto x0 = getV(verts[inds.x]);
        auto x1 = getV(verts[inds.y]);
        auto x0_ = x0 + alpha * getV(dirs[inds.x]);
        auto x1_ = x1 + alpha * getV(dirs[inds.y]);
        bv_t bv{get_bounding_box(x0, x1)};
        merge(bv, x0_);
        merge(bv, x1_);
        bv._min -= xi / 2;
        bv._max += xi / 2;
        bvs[ei] = bv;
        });
        return ret;
}

template <class element_type>
void calcLeafBvs(const double3* _vertexes, const element_type* _faces, AABB* _bvs, const int& faceNum, const int& type) {
    int numbers = faceNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs << <blockNum, threadNum >> > (_vertexes, _faces, _bvs + numbers - 1, faceNum, type);
}

template <class element_type>
void calcLeafBvs_fullCCD(const double3* _vertexes, const double3* _moveDir, const double& alpha, const element_type* _faces, AABB* _bvs, const int& faceNum, const int& type) {
    int numbers = faceNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs_ccd << <blockNum, threadNum >> > (_vertexes, _moveDir, alpha, _faces, _bvs + numbers - 1, faceNum, type);
}

void calcMChash(uint64_t* _MChash, AABB* _bvs, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcMChash << <blockNum, threadNum >> > (_MChash, _bvs, number);
}

void calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafNodes << <blockNum, threadNum >> > (_nodes, _indices, number);
}

void calcInternalNodes(Node* _nodes, const uint64_t* _MChash, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcInternalNodes << <blockNum, threadNum >> > (_nodes, _MChash, number);
}

void calcInternalAABB(const Node* _nodes, AABB* _bvs, uint32_t* flags, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    //uint32_t* flags;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&flags, (numbers-1) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(flags, 0xFFFFFFFF, sizeof(uint32_t) * (numbers - 1)));
    _calcInternalAABB << <blockNum, threadNum >> > (_nodes, _bvs, flags, numbers);
    //CUDA_SAFE_CALL(cudaFree(flags));

}

void sortBvs(const uint32_t* _indices, AABB* _bvs, AABB* _temp_bvs, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    //AABB* _temp_bvs = _tempLeafBox;
   // CUDA_SAFE_CALL(cudaMalloc((void**)&_temp_bvs, (number) * sizeof(AABB)));
    cudaMemcpy(_temp_bvs, _bvs + number - 1, sizeof(AABB) * number, cudaMemcpyDeviceToDevice);
    _sortBvs << <blockNum, threadNum >> > (_indices, _bvs + number - 1, _temp_bvs, number);
    //CUDA_SAFE_CALL(cudaFree(_temp_bvs));
}


void selfQuery_ee(const double3* _vertexes, const double3* _rest_vertexes, const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _collisonPairs, int4* _ccd_collisonPairs, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    
    _selfQuery_ee << <blockNum, threadNum >> > (_vertexes, _rest_vertexes, _edges, _bvs, _nodes, _collisonPairs, _ccd_collisonPairs, _cpNum, MatIndex, dHat, numbers);
}

void fullCCDselfQuery_ee(const double3* _vertexes, const double3* moveDir, const double& alpha, const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _ccd_collisonPairs, uint32_t* _cpNum, double dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _selfQuery_ee_ccd << <blockNum, threadNum >> > (_vertexes, moveDir, alpha, _edges, _bvs, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers);
}

void selfQuery_vf(const double3* _vertexes, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes, int4* _collisonPairs, int4* _ccd_collisonPairs, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _selfQuery_vf << <blockNum, threadNum >> > (_vertexes, _faces, _surfVerts, _bvs, _nodes, _collisonPairs, _ccd_collisonPairs, _cpNum, MatIndex, dHat, numbers);
}

void fullCCDselfQuery_vf(const double3* _vertexes, const double3* moveDir, const double& alpha, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes, int4* _ccd_collisonPairs, uint32_t* _cpNum, double dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _selfQuery_vf_ccd << <blockNum, threadNum >> > (_vertexes, moveDir, alpha, _faces, _surfVerts, _bvs, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers);
}

void lbvh::FREE_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaFree(_indices));
    CUDA_SAFE_CALL(cudaFree(_MChash));
    CUDA_SAFE_CALL(cudaFree(_nodes));
    CUDA_SAFE_CALL(cudaFree(_bvs));
    CUDA_SAFE_CALL(cudaFree(_flags));
    CUDA_SAFE_CALL(cudaFree(_tempLeafBox));
}

void lbvh::MALLOC_DEVICE_MEM(const int& number) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&_indices, (number) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MChash, (number) * sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_nodes, (2 * number - 1) * sizeof(Node)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_bvs, (2 * number - 1) * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempLeafBox, number * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_flags, (number - 1) * sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_cpNum, sizeof(uint32_t)));ye
    //CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));
}

lbvh::~lbvh() {
    //FREE_DEVICE_MEM();
}



void lbvh_f::init(double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& faceNum, const int& vertNum) {
    _faces = _mFaces;
    _surfVerts = _mSurfVert;
    _vertexes = _mVerts;
    _collisionPair = _mCollisonPairs;
    _ccd_collisionPair = _ccd_mCollisonPairs;
    _cpNum = _mcpNum;
    _MatIndex = _mMatIndex;
    face_number = faceNum;
    vert_number = vertNum;
    MALLOC_DEVICE_MEM(face_number);
}

void lbvh_e::init(double3* _mVerts, double3* _mRest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& edgeNum, const int& vertNum) {
    _rest_vertexes = _mRest_vertexes;
    _edges = _mEdges;
    _vertexes = _mVerts;
    _cpNum = _mcpNum;
    _collisionPair = _mCollisonPairs;
    _ccd_collisionPair = _ccd_mCollisonPairs;
    _MatIndex = _mMatIndex;
    edge_number = edgeNum;
    vert_number = vertNum;
    MALLOC_DEVICE_MEM(edge_number);
}

AABB* lbvh_f::getSceneSize() {
    calcLeafBvs(_vertexes, _faces, _bvs, face_number, 0);

    calcMaxBV(_bvs, _tempLeafBox, face_number);
    return _bvs;
}

double lbvh_f::Construct() {
#if 1
    calcLeafBvs(_vertexes, _faces, _bvs, face_number, 0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    scene = calcMaxBV(_bvs, _tempLeafBox, face_number);
    calcMChash(_MChash, _bvs, face_number);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + face_number);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + face_number, thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, face_number);
    calcLeafNodes(_nodes, _indices, face_number);
    calcInternalNodes(_nodes, _MChash, face_number);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calcInternalAABB(_nodes, _bvs, _flags, face_number);
#endif

    auto cudaPol = zs::cuda_exec();
    auto bvs = retrieve_bounding_volumes(cudaPol, face_number, _vertexes, _faces, 0);
    bvh.build(cudaPol, bvs);
    return 0;//time0 + time1 + time2;
}

double lbvh_f::ConstructFullCCD(const double3* moveDir, const double& alpha) {
#if 1
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _faces, _bvs, face_number, 0);
    scene = calcMaxBV(_bvs, _tempLeafBox, face_number);
    calcMChash(_MChash, _bvs, face_number);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + face_number);

    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + face_number, thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, face_number);

    calcLeafNodes(_nodes, _indices, face_number);

    calcInternalNodes(_nodes, _MChash, face_number);
    calcInternalAABB(_nodes, _bvs, _flags, face_number);
#endif

    auto cudaPol = zs::cuda_exec();
    auto bvs = retrieve_bounding_volumes(cudaPol, face_number, _vertexes, moveDir, _faces, -alpha, 0);
    bvh.build(cudaPol, bvs);
    return 0;
}

double lbvh_e::Construct() {
#if 1
    /*cudaEvent_t start, end0, end1, end2;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);

    cudaEventRecord(start);*/
    calcLeafBvs(_vertexes, _edges, _bvs, edge_number, 1);
    scene = calcMaxBV(_bvs, _tempLeafBox, edge_number);
    calcMChash(_MChash, _bvs, edge_number);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + edge_number);
    //cudaEventRecord(end0);

    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + edge_number, thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, edge_number);

    //cudaEventRecord(end1);

    calcLeafNodes(_nodes, _indices, edge_number);

    calcInternalNodes(_nodes, _MChash, edge_number);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calcInternalAABB(_nodes, _bvs, _flags, edge_number);
    //selfQuery(_vertexes, _edges, _bvs, _nodes, _collisionPair, _cpNum, edge_number);
    //cudaEventRecord(end2);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    /*float time0 = 0, time1 = 0, time2 = 0;
    cudaEventElapsedTime(&time0, start, end0);
    cudaEventElapsedTime(&time1, end0, end1);
    cudaEventElapsedTime(&time2, end1, end2);
    (cudaEventDestroy(start));
    (cudaEventDestroy(end0));
    (cudaEventDestroy(end1));
    (cudaEventDestroy(end2));*/
    //std::cout << "sort time: " << time1 << std::endl;
#endif

    auto cudaPol = zs::cuda_exec();
    auto bvs = retrieve_bounding_volumes(cudaPol, edge_number, _vertexes, _edges, 0);
    bvh.build(cudaPol, bvs);
    return 0;//time0 + time1 + time2;
    //std::cout << "generation done: " << time0 + time1 + time2 << std::endl;
}

double lbvh_e::ConstructFullCCD(const double3* moveDir, const double& alpha) {
#if 1
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _edges, _bvs, edge_number, 1);
    scene = calcMaxBV(_bvs, _tempLeafBox, edge_number);
    calcMChash(_MChash, _bvs, edge_number);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + edge_number);

    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + edge_number, thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, edge_number);

    calcLeafNodes(_nodes, _indices, edge_number);

    calcInternalNodes(_nodes, _MChash, edge_number);

    calcInternalAABB(_nodes, _bvs, _flags, edge_number);
#endif

    auto cudaPol = zs::cuda_exec();
    auto bvs = retrieve_bounding_volumes(cudaPol, edge_number, _vertexes, moveDir, _edges, -alpha, 0);
    bvh.build(cudaPol, bvs);
    return 0;
}


void lbvh_f::SelfCollitionDetect(double dHat) {
#if 0
    selfQuery_vf(_vertexes, _faces, _surfVerts, _bvs, _nodes, _collisionPair, _ccd_collisionPair, _cpNum, _MatIndex, dHat, vert_number);
#else
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();
    cudaPol(range(vert_number), [bvh = proxy<space>(bvh), _vertexes = this->_vertexes, _faces = this->_faces, _surfVerts = this->_surfVerts, _collisionPair = this->_collisionPair, _ccd_collisionPair =  this->_ccd_collisionPair, _cpNum = this->_cpNum, _MatIndex = this->_MatIndex, dHat, dh = std::sqrt(dHat)]__device__(int svi) mutable {
        auto vi = _surfVerts[svi];
        auto p = zs::vec<double, 3>{_vertexes[vi].x, _vertexes[vi].y, _vertexes[vi].z};
        bv_t bv{p - dh, p + dh};
        iter_neighbors(bvh, bv, [&](int sfi) {
            auto tri = _faces[sfi];
            if (vi != tri.x && vi != tri.y && vi != tri.z) {
                _checkPTintersection(_vertexes, vi, tri.x, tri.y, tri.z, dHat, _cpNum, _MatIndex, _collisionPair, _ccd_collisionPair);
            }
        });
    });
#endif
}

void lbvh_e::SelfCollitionDetect(double dHat) {
#if 0
    selfQuery_ee(_vertexes, _rest_vertexes, _edges, _bvs, _nodes, _collisionPair, _ccd_collisionPair, _cpNum, _MatIndex, dHat, edge_number);
#else
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();
    cudaPol(range(edge_number), [bvh = proxy<space>(bvh), _vertexes = this->_vertexes, _rest_vertexes = this->_rest_vertexes, _edges = this->_edges, _collisionPair = this->_collisionPair, _ccd_collisionPair =  this->_ccd_collisionPair, _cpNum = this->_cpNum, _MatIndex = this->_MatIndex, dHat, dh = std::sqrt(dHat)]__device__(int sei) mutable {
        auto edge = _edges[sei];
        auto ea0 = zs::vec<double, 3>{_vertexes[edge.x].x, _vertexes[edge.x].y, _vertexes[edge.x].z};
        auto ea1 = zs::vec<double, 3>{_vertexes[edge.y].x, _vertexes[edge.y].y, _vertexes[edge.y].z};
        bv_t bv{get_bounding_box(ea0, ea1)};
        bv._min -= dh;
        bv._max += dh;
        iter_neighbors(bvh, bv, [&](int sej) {
            auto oedge = _edges[sej];
            if (edge.x != oedge.x && edge.x != oedge.y && edge.y != oedge.x && edge.y != oedge.y && sei > sej) {
                _checkEEintersection(_vertexes, _rest_vertexes, edge.x, edge.y, oedge.x, oedge.y, dHat, _cpNum, _MatIndex, _collisionPair, _ccd_collisionPair, 0);
            }
        });
    });
#endif
}


void lbvh_f::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha) {
#if 0
    fullCCDselfQuery_vf(_vertexes, moveDir, alpha, _faces, _surfVerts, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat, vert_number);
#else
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();
    cudaPol(range(vert_number), [bvh = proxy<space>(bvh), moveDir, alpha, _vertexes = this->_vertexes, _faces = this->_faces, _surfVerts = this->_surfVerts, _collisionPair = this->_collisionPair, _ccd_collisionPair =  this->_ccd_collisionPair, _cpNum = this->_cpNum, _MatIndex = this->_MatIndex, dHat, dh = std::sqrt(dHat), xi = 0]__device__(int svi) mutable {
        auto vi = _surfVerts[svi];
        auto p = zs::vec<double, 3>{_vertexes[vi].x, _vertexes[vi].y, _vertexes[vi].z};
        auto dp = -alpha * zs::vec<double, 3>{moveDir[vi].x, moveDir[vi].y, moveDir[vi].z};
        bv_t bv{get_bounding_box(p, p + dp)};
        bv._min -= (xi / 2 + dh);
        bv._max += (xi / 2 + dh);
        iter_neighbors(bvh, bv, [&](int sfi) {
            auto tri = _faces[sfi];
            if (vi != tri.x && vi != tri.y && vi != tri.z) {
                _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-vi - 1, tri.x, tri.y, tri.z);
            }
        });
    });
#endif
}

void lbvh_e::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha) {
#if 0
    fullCCDselfQuery_ee(_vertexes, moveDir, alpha, _edges, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat, edge_number);
#else
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();
    cudaPol(range(edge_number), [bvh = proxy<space>(bvh), moveDir, alpha, _vertexes = this->_vertexes, _rest_vertexes = this->_rest_vertexes, _edges = this->_edges, _collisionPair = this->_collisionPair, _ccd_collisionPair =  this->_ccd_collisionPair, _cpNum = this->_cpNum, _MatIndex = this->_MatIndex, dHat, dh = std::sqrt(dHat), xi = 0]__device__(int sei) mutable {
        auto edge = _edges[sei];
        auto ea0 = zs::vec<double, 3>{_vertexes[edge.x].x, _vertexes[edge.x].y, _vertexes[edge.x].z};
        auto dea0 = -alpha * zs::vec<double, 3>{moveDir[edge.x].x, moveDir[edge.x].y, moveDir[edge.x].z};
        auto ea1 = zs::vec<double, 3>{_vertexes[edge.y].x, _vertexes[edge.y].y, _vertexes[edge.y].z};
        auto dea1 = -alpha * zs::vec<double, 3>{moveDir[edge.y].x, moveDir[edge.y].y, moveDir[edge.y].z};
        bv_t bv{get_bounding_box(ea0, ea1)};
        merge(bv, ea0 + dea0);
        merge(bv, ea1 + dea1);
        bv._min -= (xi / 2 + dh);
        bv._max += (xi / 2 + dh);
        iter_neighbors(bvh, bv, [&](int sej) {
            auto oedge = _edges[sej];
            if (edge.x != oedge.x && edge.x != oedge.y && edge.y != oedge.x && edge.y != oedge.y && sei > sej) {
                _ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(edge.x, edge.y, oedge.x, oedge.y);
            }
        });
    });
#endif

}



//#include <cstdio>
//#include <cstdlib>
//#include <vector>
//
//#include <cuda_runtime.h>
//#include <cusolverDn.h>
//#include <random>
//
//#include <cstdlib>
//
//int main2() {
//    cusolverDnHandle_t cusolverH = NULL;
//    cudaStream_t stream = NULL;
//
//    const int m = 12;
//    const int lda = m;
//    /*
//     *       | 3.5 0.5 0.0 |
//     *   A = | 0.5 3.5 0.0 |
//     *       | 0.0 0.0 2.0 |
//     *
//     */
//    std::vector<double> A;// = { 3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0 };
//    //const std::vector<double> lambda = { 2.0, 3.0, 4.0 };
//    for (int i = 0;i < m;i++) {
//        for (int j = 0;j < m;j++) {
//            A.push_back((double)rand() / RAND_MAX);
//        }
//    }
//
//    std::vector<double> V(lda * m, 0); // eigenvectors
//    std::vector<double> W(m, 0);       // eigenvalues
//
//    double* d_A = nullptr;
//    double* d_W = nullptr;
//    int* d_info = nullptr;
//
//    int info = 0;
//
//    int lwork = 0;            /* size of workspace */
//    double* d_work = nullptr; /* device workspace*/
//
//    std::printf("A = (matlab base-1)\n");
//    //print_matrix(m, m, A.data(), lda);
//    std::printf("=====\n");
//    
//    cudaEvent_t start, end0;
//    cudaEventCreate(&start);
//    cudaEventCreate(&end0);
//
//    
//    /* step 1: create cusolver handle, bind a stream */
//    (cusolverDnCreate(&cusolverH));
//
//    (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//    (cusolverDnSetStream(cusolverH, stream));
//
//    (cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * A.size()));
//    (cudaMalloc(reinterpret_cast<void**>(&d_W), sizeof(double) * W.size()));
//    (cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
//
//    (
//        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
//
//    // step 3: query working space of syevd
//    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
//    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//    cudaEventRecord(start);
//    (cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork));
//
//    (cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(double) * lwork));
//
//    // step 4: compute spectrum
//    (
//        cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, d_info));
//    cudaEventRecord(end0);
//    (
//        cudaMemcpyAsync(V.data(), d_A, sizeof(double) * V.size(), cudaMemcpyDeviceToHost, stream));
//    (
//        cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(), cudaMemcpyDeviceToHost, stream));
//    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
//
//    (cudaStreamSynchronize(stream));
//
//
//    
//    CUDA_SAFE_CALL(cudaDeviceSynchronize());
//
//    float time0 = 0, time1 = 0, time2 = 0;
//    cudaEventElapsedTime(&time0, start, end0);
//
//    (cudaEventDestroy(start));
//    (cudaEventDestroy(end0));
//
//    std::printf("after syevd: info = %d  %f\n", info, time0);
//    if (0 > info) {
//        std::printf("%d-th parameter is wrong \n", -info);
//        exit(1);
//    }
//
//    std::printf("eigenvalue = (matlab base-1), ascending order\n");
//    int idx = 1;
//    for (auto const& i : W) {
//        std::printf("W[%i] = %E\n", idx, i);
//        idx++;
//    }
//
//
//    (cudaFree(d_A));
//    (cudaFree(d_W));
//    (cudaFree(d_info));
//    (cudaFree(d_work));
//
//    (cusolverDnDestroy(cusolverH));
//
//    (cudaStreamDestroy(stream));
//
//    (cudaDeviceReset());
//
//    return EXIT_SUCCESS;
//}
