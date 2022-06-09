#include "GIPC.cuh"
#include "cuda_tools.h"
#include "GIPC_PDerivative.cuh"
#include "fem_parameters.h"
#include "ACCD.cuh"
#include "femEnergy.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
// #include "FrictionUtils.cuh"
#include <fstream>
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/matrix/Eigen.hpp"

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
inline uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__
inline uint32_t hash_code(int type, double x, double y, double z, double resolution = 1024) noexcept
{
    x = __m_min(__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __m_min(__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __m_min(__m_max(z * resolution, 0.0), resolution - 1.0);

    //const uint32_t xx = expand_bits(static_cast<uint32_t>(x));
    //const uint32_t yy = expand_bits(static_cast<uint32_t>(y));
    //const uint32_t zz = expand_bits(static_cast<uint32_t>(z));
    //
    if (type == 0) {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024) + static_cast<uint32_t>(x);
    }
    else if (type == 1) {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024) + static_cast<uint32_t>(x);
    }
    else if (type == 2) {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024) + static_cast<uint32_t>(y);
    }
    else if (type == 3) {
        return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024) + static_cast<uint32_t>(y);
    }
    else if (type == 4) {
        return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024) + static_cast<uint32_t>(z);
    }
    else {
        return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024) + static_cast<uint32_t>(z);
    }
    //std::uint32_t mchash = (((static_cast<std::uint32_t>(z) * 1024) + static_cast<std::uint32_t>(y)) * 1024) + static_cast<std::uint32_t>(x);//((xx << 2) + (yy << 1) + zz);
    //return mchash;
}

__global__
void _calcTetMChash(uint64_t* _MChash, const double3* _vertexes, uint4* tets, const AABB* _MaxBv, const uint32_t* sortMapVertIndex, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;

    tets[idx].x = sortMapVertIndex[tets[idx].x];
    tets[idx].y = sortMapVertIndex[tets[idx].y];
    tets[idx].z = sortMapVertIndex[tets[idx].z];
    tets[idx].w = sortMapVertIndex[tets[idx].w];

    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x, (*_MaxBv).upper.y - (*_MaxBv).lower.y, (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__add(__GEIGEN__::__add(_vertexes[tets[idx].x], _vertexes[tets[idx].y]), __GEIGEN__::__add(_vertexes[tets[idx].z], _vertexes[tets[idx].w])), 0.25);
    double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x, centerP.y - (*_MaxBv).lower.y, centerP.z - (*_MaxBv).lower.z);

    int type = 0;
    if (SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z) {
        type = 0;
    }
    else if (SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y) {
        type = 1;
    }
    else if (SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x) {
        type = 2;
    }
    else if (SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z) {
        type = 3;
    }
    else if (SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y) {
        type = 4;
    }
    else {
        type = 5;
    }

    //printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = hash_code(type, offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %d\n", mc64);
    _MChash[idx] = mc64;
}

__global__
void _updateVertexes(double3* o_vertexes, const double3* _vertexes, double* tempM, const double* mass, __GEIGEN__::Matrix3x3d* tempCons, int* tempBtype, const __GEIGEN__::Matrix3x3d* cons, const int* bType, const uint32_t* sortIndex, uint32_t* sortMapIndex, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    o_vertexes[idx] = _vertexes[sortIndex[idx]];
    tempM[idx] = mass[sortIndex[idx]];
    tempCons[idx] = cons[sortIndex[idx]];
    sortMapIndex[sortIndex[idx]] = idx;
    tempBtype[idx] = bType[sortIndex[idx]];
    //printf("original idx: %d        new idx: %d\n", sortIndex[idx], idx);
}

__global__
void _updateTetrahedras(uint4* o_tetrahedras, uint4* tetrahedras, double* tempV, const double* volum, __GEIGEN__::Matrix3x3d* tempDmInverse, const __GEIGEN__::Matrix3x3d* dmInverse, const uint32_t* sortTetIndex, const uint32_t* sortMapVertIndex, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    //tetrahedras[idx].x = sortMapVertIndex[tetrahedras[idx].x];
    //tetrahedras[idx].y = sortMapVertIndex[tetrahedras[idx].y];
    //tetrahedras[idx].z = sortMapVertIndex[tetrahedras[idx].z];
    //tetrahedras[idx].w = sortMapVertIndex[tetrahedras[idx].w];
    o_tetrahedras[idx] = tetrahedras[sortTetIndex[idx]];
    tempV[idx] = volum[sortTetIndex[idx]];
    tempDmInverse[idx] = dmInverse[sortTetIndex[idx]];
}

__global__
void _calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x, (*_MaxBv).upper.y - (*_MaxBv).lower.y, (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP = _vertexes[idx];
    double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x, centerP.y - (*_MaxBv).lower.y, centerP.z - (*_MaxBv).lower.z);
    int type = 0;
    if (SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z) {
        type = 0;
    }
    else if (SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y) {
        type = 1;
    }
    else if (SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x) {
        type = 2;
    }
    else if (SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z) {
        type = 3;
    }
    else if (SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y) {
        type = 4;
    }
    else {
        type = 5;
    }

    //printf("minSize %f     %f     %f\n", SceneSize.x, SceneSize.y, SceneSize.z);
    uint64_t mc32 = hash_code(type, offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    //printf("morton code %lld\n", mc64);
    _MChash[idx] = mc64;
}

__global__
void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 tempMove = _double3Dim[idx];

    double temp = __GEIGEN__::__norm(tempMove);//__m_max(__m_max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp, i);
        temp = __m_max(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp, i);
            temp = __m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__
void _reduct_min_double(double* _double1Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp, i);
        temp = __m_min(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp, i);
            temp = __m_min(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__
void _reduct_M_double2(double2* _double2Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double2 temp = _double2Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp.x, i);
        double tempMax = __shfl_down(temp.y, i);
        temp.x = __m_max(temp.x, tempMin);
        temp.y = __m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp.x, i);
            double tempMax = __shfl_down(temp.y, i);
            temp.x = __m_max(temp.x, tempMin);
            temp.y = __m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _double2Dim[blockIdx.x] = temp;
    }
}

__global__
void _reduct_max_double(double* _double1Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMax = __shfl_down(temp, i);
        temp = __m_max(temp, tempMax);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMax = __shfl_down(temp, i);
            temp = __m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _double1Dim[blockIdx.x] = temp;
    }
}

__device__
double __cal_Barrier_energy(const double3* _vertexes, const double3* _rest_vertexes, int4 MMCVIDI, double _Kappa, double _dHat) {
    double dHat_sqrt = sqrt(_dHat);
    double dHat = _dHat;
    double Kappa = _Kappa;
    using namespace zs;
    using T = double;
    constexpr double xi2 = 0.0;
    const double activeGap2 = dHat;
    auto getV = [](const double3& v) {
        return zs::vec<double, 3>{v.x, v.y, v.z};
    };
    auto get_mollifier =
          [&getV, _vertexes, _rest_vertexes] __device__(int id0, int id1, int id2, int id3) {
            auto ea0Rest = getV(_rest_vertexes[id0]);
            auto ea1Rest = getV(_rest_vertexes[id1]);
            auto eb0Rest = getV(_rest_vertexes[id2]);
            auto eb1Rest = getV(_rest_vertexes[id3]);
            T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
            auto ea0 = getV(_vertexes[id0]);
            auto ea1 = getV(_vertexes[id1]);
            auto eb0 = getV(_vertexes[id2]);
            auto eb1 = getV(_vertexes[id3]);
            return mollifier_ee(ea0, ea1, eb0, eb1, epsX);
    };
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {   // ee
            auto ea0 = getV(_vertexes[MMCVIDI.x]);
            auto ea1 = getV(_vertexes[MMCVIDI.y]);
            auto eb0 = getV(_vertexes[MMCVIDI.z]);
            auto eb1 = getV(_vertexes[MMCVIDI.w]);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            return barrier(dist2 - xi2, activeGap2, Kappa);
        }
        else {  // eem
            MMCVIDI.w = -MMCVIDI.w - 1;
            auto ea0 = getV(_vertexes[MMCVIDI.x]);
            auto ea1 = getV(_vertexes[MMCVIDI.y]);
            auto eb0 = getV(_vertexes[MMCVIDI.z]);
            auto eb1 = getV(_vertexes[MMCVIDI.w]);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            return get_mollifier(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w) * barrier(dist2 - xi2, activeGap2, Kappa);
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {    // ppm
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                int ppm[4] = {MMCVIDI.x, MMCVIDI.z, MMCVIDI.y, MMCVIDI.w};
                auto ea0 = getV(_vertexes[ppm[0]]);
                auto ea1 = getV(_vertexes[ppm[1]]);
                auto eb0 = getV(_vertexes[ppm[2]]);
                auto eb1 = getV(_vertexes[ppm[3]]);
                auto dist2 = dist2_pp(ea0, eb0);
                return get_mollifier(ppm[0], ppm[1], ppm[2], ppm[3]) * barrier(dist2 - xi2, activeGap2, Kappa);
            }
            else {  // pp
                auto x0 = getV(_vertexes[v0I]);
                auto x1 = getV(_vertexes[MMCVIDI.y]);
                auto dist2 = dist2_pp(x0, x1);
                return barrier(dist2 - xi2, activeGap2, Kappa);
            }
        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {    // pem
                MMCVIDI.y = -MMCVIDI.y - 1;
                //MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                int pem[4] = {MMCVIDI.x, MMCVIDI.w, MMCVIDI.y, MMCVIDI.z};
                auto ea0 = getV(_vertexes[pem[0]]);
                auto ea1 = getV(_vertexes[pem[1]]);
                auto eb0 = getV(_vertexes[pem[2]]);
                auto eb1 = getV(_vertexes[pem[3]]);
                auto dist2 = dist2_pe(ea0, eb0, eb1);
                return get_mollifier(pem[0], pem[1], pem[2], pem[3]) * barrier(dist2 - xi2, activeGap2, Kappa);
            }
            else {  // pe
                auto p = getV(_vertexes[v0I]);
                auto e0 = getV(_vertexes[MMCVIDI.y]);
                auto e1 = getV(_vertexes[MMCVIDI.z]);
                auto dist2 = dist2_pe(p, e0, e1);
                return barrier(dist2 - xi2, activeGap2, Kappa);
            }
        }
        else {  // pt
            auto p = getV(_vertexes[v0I]);
            auto t0 = getV(_vertexes[MMCVIDI.y]);
            auto t1 = getV(_vertexes[MMCVIDI.z]);
            auto t2 = getV(_vertexes[MMCVIDI.w]);
            auto dist2 = dist2_pt(p, t0, t1, t2);
            return barrier(dist2 - xi2, activeGap2, Kappa);
        }
    }
}

__device__
bool segTriIntersect(const double3& ve0, const double3& ve1,
    const double3& vt0, const double3& vt1, const double3& vt2)
{

    //printf("check for tri and lines\n");

    __GEIGEN__::Matrix3x3d coefMtr;
    double3 col0 = __GEIGEN__::__minus(vt1, vt0);
    double3 col1 = __GEIGEN__::__minus(vt2, vt0);
    double3 col2 = __GEIGEN__::__minus(ve0, ve1);

    __GEIGEN__::__set_Mat_val_column(coefMtr, col0, col1, col2);

    double3 n = __GEIGEN__::__v_vec_cross(col0, col1);
    if (__GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve0, vt0)) * __GEIGEN__::__v_vec_dot(n, __GEIGEN__::__minus(ve1, vt0)) > 0) {
        return false;
    }

    double det = __GEIGEN__::__Determiant(coefMtr);

    if (det == 0) {
        return false;
    }

    __GEIGEN__::Matrix3x3d D1, D2, D3;
    double3 b = __GEIGEN__::__minus(ve0, vt0);

    __GEIGEN__::__set_Mat_val_column(D1, b, col1, col2);
    __GEIGEN__::__set_Mat_val_column(D2, col0, b, col2);
    __GEIGEN__::__set_Mat_val_column(D3, col0, col1, b);

    double uvt[3];
    uvt[0] = __GEIGEN__::__Determiant(D1) / det;
    uvt[1] = __GEIGEN__::__Determiant(D2) / det;
    uvt[2] = __GEIGEN__::__Determiant(D3) / det;

    if (uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0 && uvt[2] <= 1.0) {
        return true;
    }
    else {
        return false;
    }
}

__device__ __host__
inline bool _overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

__device__
double _selfConstraintVal(const double3* vertexes, const int4& active) {
    double val;
    if (active.x >= 0) {
        if (active.w >= 0) {
            _d_EE(vertexes[active.x], vertexes[active.y], vertexes[active.z], vertexes[active.w], val);
        }
        else {
            _d_EE(vertexes[active.x], vertexes[active.y], vertexes[active.z], vertexes[-active.w - 1], val);
        }
    }
    else {
        if (active.z < 0) {
            if (active.y < 0) {
                _d_PP(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            }
            else {
                _d_PP(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        }
        else if (active.w < 0) {
            if (active.y < 0) {
                _d_PE(vertexes[-active.x - 1], vertexes[-active.y - 1], vertexes[active.z], val);
            }
            else {
                _d_PE(vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], val);
            }
        }
        else {
            _d_PT(vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], vertexes[active.w], val);
        }
    }
    return val;
}

__device__
double _computeInjectiveStepSize_3d(const double3* verts, const double3* mv, const uint32_t& v0, const uint32_t& v1, const uint32_t& v2, const uint32_t& v3, double ratio, double errorRate) {

    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;


    x1 = verts[v0].x;
    x2 = verts[v1].x;
    x3 = verts[v2].x;
    x4 = verts[v3].x;

    y1 = verts[v0].y;
    y2 = verts[v1].y;
    y3 = verts[v2].y;
    y4 = verts[v3].y;

    z1 = verts[v0].z;
    z2 = verts[v1].z;
    z3 = verts[v2].z;
    z4 = verts[v3].z;

    int _3Fii0 = v0 * 3;
    int _3Fii1 = v1 * 3;
    int _3Fii2 = v2 * 3;
    int _3Fii3 = v3 * 3;

    p1 = -mv[v0].x;
    p2 = -mv[v1].x;
    p3 = -mv[v2].x;
    p4 = -mv[v3].x;

    q1 = -mv[v0].y;
    q2 = -mv[v1].y;
    q3 = -mv[v2].y;
    q4 = -mv[v3].y;

    r1 = -mv[v0].z;
    r2 = -mv[v1].z;
    r3 = -mv[v2].z;
    r4 = -mv[v3].z;

    a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 - r1 * p2 * q3 + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 - q1 * p2 * r4 + q1 * r2 * p4 + r1 * p2 * q4 - r1 * q2 * p4 - p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4 - q1 * r3 * p4 - r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4 - q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
    b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 - z1 * p2 * q3 + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 - y2 * p1 * r3 + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 - x3 * q1 * r2 + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 - z3 * p1 * q2 + z3 * q1 * p2 + x1 * q2 * r4 - x1 * r2 * q4 - y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4 - z1 * q2 * p4 - x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4 - z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 - y4 * p1 * r2 + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 - x1 * q3 * r4 + x1 * r3 * q4 + y1 * p3 * r4 - y1 * r3 * p4 - z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4 - x3 * r1 * q4 - y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4 - x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 - z4 * p1 * q3 + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 - y2 * p3 * r4 + y2 * r3 * p4 + z2 * p3 * q4 - z2 * q3 * p4 - x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4 - y3 * r2 * p4 - z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3 - y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
    c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 + y1 * x2 * r3 - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 - z1 * x2 * q3 + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 - x2 * y3 * r1 + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 - z2 * x3 * q1 + z2 * y3 * p1 + x1 * y2 * r4 - x1 * z2 * q4 - x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4 + y1 * z2 * p4 + y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4 - z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 - y2 * x4 * r1 + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 - x1 * y3 * r4 + x1 * z3 * q4 + x1 * y4 * r3 - x1 * z4 * q3 + y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3 + y1 * z4 * p3 - z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3 - x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 - z3 * x4 * q1 + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 - x2 * y4 * r3 + x2 * z4 * q3 - y2 * x3 * r4 + y2 * z3 * p4 + y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4 - z2 * y3 * p4 - z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2 - y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
    d = (ratio) * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3 + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4 + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4 - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4 - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);


    //printf("a b c d:   %f  %f  %f  %f     %f     %f,    id0, id1, id2, id3:  %d  %d  %d  %d\n", a, b, c, d, ratio, errorRate, v0, v1, v2, v3);
    if (abs(a) <= errorRate) {
        if (abs(b) <= errorRate)
            if (abs(c) <= errorRate) {
                t = 1;
            }
            else {
                t = -d / c;
            }
        else {
            double desc = c * c - 4 * b * d;
            if (desc > 0) {
                t = (-c - sqrt(desc)) / (2 * b);
                if (t < 0)
                    t = (-c + sqrt(desc)) / (2 * b);
            }
            else
                t = 1;
        }
    }
    else {
        double results[3];
        int number = 0;
        __GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, results, number, errorRate);

        t = 1;
        for (int index = 0;index < number;index++) {
            if (results[index] > 0 && results[index] < t) {
                t = results[index];
            }
        }
    }
    if (t < 0)
        t = 1;
    return t;
}


__global__
void _calBarrierHessian(const double3* _vertexes, const double3* _rest_vertexes, const int4* _collisionPair, __GEIGEN__::Matrix12x12d* H12x12, __GEIGEN__::Matrix9x9d* H9x9,
    __GEIGEN__::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum, int* matIndex, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;

    using namespace zs;
    using T = double;
    using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
    constexpr double xi2 = 0.0;
    const double activeGap2 = dHat;
    auto getV = [](const double3& v) {
        return zs::vec<double, 3>{v.x, v.y, v.z};
    };
    auto get_mollifier =
          [&getV, _vertexes, _rest_vertexes] __device__(int id0, int id1, int id2, int id3) {
            auto ea0Rest = getV(_rest_vertexes[id0]);
            auto ea1Rest = getV(_rest_vertexes[id1]);
            auto eb0Rest = getV(_rest_vertexes[id2]);
            auto eb1Rest = getV(_rest_vertexes[id3]);
            T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
            auto ea0 = getV(_vertexes[id0]);
            auto ea1 = getV(_vertexes[id1]);
            auto eb0 = getV(_vertexes[id2]);
            auto eb1 = getV(_vertexes[id3]);
            return zs::make_tuple(mollifier_ee(ea0, ea1, eb0, eb1, epsX),
                                  mollifier_grad_ee(ea0, ea1, eb0, eb1, epsX),
                                  mollifier_hess_ee(ea0, ea1, eb0, eb1, epsX));
    };
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            auto ea0 = getV(_vertexes[MMCVIDI.x]);
            auto ea1 = getV(_vertexes[MMCVIDI.y]);
            auto eb0 = getV(_vertexes[MMCVIDI.z]);
            auto eb1 = getV(_vertexes[MMCVIDI.w]);
            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2) {
              dist2 = xi2 + 1e-3 * dHat;
              printf("dist already smaller than xi!\n");
            }
            auto barrierDistGrad =
                barrier_gradient(dist2 - xi2, activeGap2, Kappa);
            // hessian
            auto eeHess = dist_hess_ee(ea0, ea1, eb0, eb1);
            auto eeGrad_ = Vec12View{eeGrad.data()};
            eeHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, Kappa) *
                          dyadic_prod(eeGrad_, eeGrad_) +
                      barrierDistGrad * eeHess);
            // make pd
            make_pd(eeHess);
            __GEIGEN__::Matrix12x12d Hessian;
            for (int i = 0; i != 12; ++i)
            for (int j = 0; j != 12; ++j)
                Hessian.m[i][j] = eeHess(i, j);
            int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);
            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
        else {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            auto ea0 = getV(_vertexes[MMCVIDI.x]);
            auto ea1 = getV(_vertexes[MMCVIDI.y]);
            auto eb0 = getV(_vertexes[MMCVIDI.z]);
            auto eb1 = getV(_vertexes[MMCVIDI.w]);

            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2) 
                printf("dist already smaller than xi!\n");
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, Kappa);
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, Kappa);
            auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, Kappa);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] = get_mollifier(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            using HessT = RM_CVREF_T(mollifierHessEE);
            using GradT = zs::vec<T, 12>;

            // hessian
            auto eeGrad_ = Vec12View{eeGrad.data()};
            auto eemHess = barrierDist2 * mollifierHessEE +
                       barrierDistGrad * (dyadic_prod(Vec12View{mollifierGradEE.data()}, eeGrad_) + dyadic_prod(eeGrad_, Vec12View{mollifierGradEE.data()}));

            auto eeHess = dist_hess_ee(ea0, ea1, eb0, eb1);
            eeHess = (barrierDistHess * dyadic_prod(eeGrad_, eeGrad_) +
                  barrierDistGrad * eeHess);
            eemHess += mollifierEE * eeHess;
            // make pd
            make_pd(eemHess);
            
            __GEIGEN__::Matrix12x12d Hessian;
            for (int i = 0; i != 12; ++i)
            for (int j = 0; j != 12; ++j)
                Hessian.m[i][j] = eemHess(i, j);
            int Hidx = matIndex[idx];
            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {
                // ppm: ----
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                int ppm[4] = {MMCVIDI.x, MMCVIDI.z, MMCVIDI.y, MMCVIDI.w};
                auto ea0 = getV(_vertexes[ppm[0]]);
                auto ea1 = getV(_vertexes[ppm[1]]);
                auto eb0 = getV(_vertexes[ppm[2]]);
                auto eb1 = getV(_vertexes[ppm[3]]);
                
                auto ppGrad = dist_grad_pp(ea0, eb0);
                auto dist2 = dist2_pp(ea0, eb0);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDist2 = barrier(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, Kappa);
                auto [mollifierEE, mollifierGradEE, mollifierHessEE] = get_mollifier(ppm[0], ppm[1], ppm[2], ppm[3]);
                using GradT = zs::vec<T, 12>;

                // hessian
                auto extendedPPGrad = GradT::zeros();
                for (int d = 0; d != 3; ++d) {
                    extendedPPGrad(d) = barrierDistGrad * ppGrad(0, d);
                    extendedPPGrad(6 + d) = barrierDistGrad * ppGrad(1, d);
                }
                auto ppmHess =
                    barrierDist2 * mollifierHessEE +
                    dyadic_prod(Vec12View{mollifierGradEE.data()}, extendedPPGrad) +
                    dyadic_prod(extendedPPGrad, Vec12View{mollifierGradEE.data()});

                auto ppHess = dist_hess_pp(ea0, eb0);
                auto ppGrad_ = Vec6View{ppGrad.data()};

                ppHess = (barrierDistHess * dyadic_prod(ppGrad_, ppGrad_) +
                        barrierDistGrad * ppHess);
                for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) {
                    ppmHess(0 + i, 0 + j) += mollifierEE * ppHess(0 + i, 0 + j);
                    ppmHess(0 + i, 6 + j) += mollifierEE * ppHess(0 + i, 3 + j);
                    ppmHess(6 + i, 0 + j) += mollifierEE * ppHess(3 + i, 0 + j);
                    ppmHess(6 + i, 6 + j) += mollifierEE * ppHess(3 + i, 3 + j);
                }
                // make pd
                make_pd(ppmHess);

                __GEIGEN__::Matrix12x12d Hessian;
                for (int i = 0; i != 12; ++i)
                    for (int j = 0; j != 12; ++j)
                        Hessian.m[i][j] = ppmHess(i, j);
                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 4, 1);

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(ppm[0], ppm[1], ppm[2], ppm[3]);

            }
            else {
                // pp: -+__
                auto x0 = getV(_vertexes[v0I]);
                auto x1 = getV(_vertexes[MMCVIDI.y]);
                auto ppGrad = dist_grad_pp(x0, x1);
                auto dist2 = dist2_pp(x0, x1);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                // hessian
                auto ppHess = dist_hess_pp(x0, x1);
                auto ppGrad_ = Vec6View{ppGrad.data()};
                ppHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, Kappa) *
                          dyadic_prod(ppGrad_, ppGrad_) + barrierDistGrad * ppHess);
                // make pd
                make_pd(ppHess);

                __GEIGEN__::Matrix6x6d Hessian;
                for (int i = 0; i != 6; ++i)
                for (int j = 0; j != 6; ++j)
                    Hessian.m[i][j] = ppHess(i, j);

                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 2, 1);

                H6x6[Hidx] = Hessian;
                D2Index[Hidx] = make_uint2(v0I, MMCVIDI.y);
            }
            //BH.D2Index.emplace_back(v0I, MMCVIDI[1]);
            //BH.H6x6.emplace_back(hessian);
        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                // pem: --+-
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                
                int pem[4] = {MMCVIDI.x, MMCVIDI.w, MMCVIDI.y, MMCVIDI.z};
                auto ea0 = getV(_vertexes[pem[0]]);
                auto ea1 = getV(_vertexes[pem[1]]);
                auto eb0 = getV(_vertexes[pem[2]]);
                auto eb1 = getV(_vertexes[pem[3]]);

                auto peGrad = dist_grad_pe(ea0, eb0, eb1);
                auto dist2 = dist2_pe(ea0, eb0, eb1);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDist2 = barrier(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, Kappa);
                auto [mollifierEE, mollifierGradEE, mollifierHessEE] = get_mollifier(pem[0], pem[1], pem[2], pem[3]);
                using GradT = zs::vec<T, 12>;

                // hessian
                auto extendedPEGrad = GradT::zeros();
                for (int d = 0; d != 3; ++d) {
                    extendedPEGrad(d) = barrierDistGrad * peGrad(0, d);
                    extendedPEGrad(6 + d) = barrierDistGrad * peGrad(1, d);
                    extendedPEGrad(9 + d) = barrierDistGrad * peGrad(2, d);
                }
                auto pemHess =
                    barrierDist2 * mollifierHessEE +
                    dyadic_prod(Vec12View{mollifierGradEE.data()}, extendedPEGrad) +
                    dyadic_prod(extendedPEGrad, Vec12View{mollifierGradEE.data()});

                auto peHess = dist_hess_pe(ea0, eb0, eb1);
                auto peGrad_ = Vec9View{peGrad.data()};

                peHess = (barrierDistHess * dyadic_prod(peGrad_, peGrad_) +
                        barrierDistGrad * peHess);
                for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j) {
                    pemHess(0 + i, 0 + j) += mollifierEE * peHess(0 + i, 0 + j);
                    //
                    pemHess(0 + i, 6 + j) += mollifierEE * peHess(0 + i, 3 + j);
                    pemHess(0 + i, 9 + j) += mollifierEE * peHess(0 + i, 6 + j);
                    //
                    pemHess(6 + i, 0 + j) += mollifierEE * peHess(3 + i, 0 + j);
                    pemHess(9 + i, 0 + j) += mollifierEE * peHess(6 + i, 0 + j);
                    //
                    pemHess(6 + i, 6 + j) += mollifierEE * peHess(3 + i, 3 + j);
                    pemHess(6 + i, 9 + j) += mollifierEE * peHess(3 + i, 6 + j);
                    pemHess(9 + i, 6 + j) += mollifierEE * peHess(6 + i, 3 + j);
                    pemHess(9 + i, 9 + j) += mollifierEE * peHess(6 + i, 6 + j);
                }

                // make pd
                make_pd(pemHess);
                __GEIGEN__::Matrix12x12d Hessian;
                for (int i = 0; i != 12; ++i)
                    for (int j = 0; j != 12; ++j)
                        Hessian.m[i][j] = pemHess(i, j);

                int Hidx = matIndex[idx];//int Hidx = Hidx = atomicAdd(_cpNum + 4, 1);

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(pem[0], pem[1], pem[2], pem[3]);
            }
            else {
                // pe: -++_
                auto p = getV(_vertexes[v0I]);
                auto e0 = getV(_vertexes[MMCVIDI.y]);
                auto e1 = getV(_vertexes[MMCVIDI.z]);
                auto peGrad = dist_grad_pe(p, e0, e1);
                auto dist2 = dist2_pe(p, e0, e1);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                // hessian
                auto peHess = dist_hess_pe(p, e0, e1);
                auto peGrad_ = Vec9View{peGrad.data()};
                peHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, Kappa) *
                          dyadic_prod(peGrad_, peGrad_) +
                      barrierDistGrad * peHess);
                // make pd
                make_pd(peHess);
                __GEIGEN__::Matrix9x9d Hessian;
                for (int i = 0; i != 9; ++i)
                for (int j = 0; j != 9; ++j)
                    Hessian.m[i][j] = peHess(i, j);

                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 3, 1);

                H9x9[Hidx] = Hessian;
                D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            }
        }
        else {
            // pt: -+++
            auto p = getV(_vertexes[v0I]);
            auto t0 = getV(_vertexes[MMCVIDI.y]);
            auto t1 = getV(_vertexes[MMCVIDI.z]);
            auto t2 = getV(_vertexes[MMCVIDI.w]);
            auto ptGrad = dist_grad_pt(p, t0, t1, t2);
            auto dist2 = dist2_pt(p, t0, t1, t2);
            if (dist2 < xi2) {
                printf("dist already smaller than xi!\n");
            }
            auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, Kappa);
            // hessian
            auto ptHess = dist_hess_pt(p, t0, t1, t2);
            auto ptGrad_ = Vec12View{ptGrad.data()};
            ptHess = (zs::barrier_hessian(dist2 - xi2, activeGap2, Kappa) *
                      dyadic_prod(ptGrad_, ptGrad_) + barrierDistGrad * ptHess);
            // make pd
            make_pd(ptHess);
            __GEIGEN__::Matrix12x12d Hessian;
            for (int i = 0; i != 12; ++i)
            for (int j = 0; j != 12; ++j)
                Hessian.m[i][j] = ptHess(i, j);

            int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
}


__global__
void _calSelfCloseVal(const double3* _vertexes, const int4* _collisionPair, int4* _close_collisionPair, double* _close_collisionVal,
    uint32_t* _close_cpNum, double dTol, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dist2 = _selfConstraintVal(_vertexes, MMCVIDI);
    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx] = dist2;
    }

}

__global__
void _checkSelfCloseVal(const double3* _vertexes, int* _isChange, int4* _close_collisionPair, double* _close_collisionVal, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _close_collisionPair[idx];
    double dist2 = _selfConstraintVal(_vertexes, MMCVIDI);
    if (dist2 < _close_collisionVal[idx]) {
        *_isChange = 1;
    }

}


__global__
void _reduct_MSelfDist(const double3* _vertexes, int4* _collisionPairs, double2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    int4 MMCVIDI = _collisionPairs[idx];
    double tempv = _selfConstraintVal(_vertexes, MMCVIDI);
    double2 temp = make_double2(1.0 / tempv, tempv);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp.x, i);
        double tempMax = __shfl_down(temp.y, i);
        temp.x = __m_max(temp.x, tempMin);
        temp.y = __m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp.x, i);
            double tempMax = __shfl_down(temp.y, i);
            temp.x = __m_max(temp.x, tempMin);
            temp.y = __m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}


__global__
void _calBarrierGradient(const double3* _vertexes, const double3* _rest_vertexes, const const int4* _collisionPair, double3* _gradient, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;

    using namespace zs;
    using T = double;
    constexpr double xi2 = 0.0;
    const double activeGap2 = dHat;
    auto getV = [](const double3& v) {
        return zs::vec<double, 3>{v.x, v.y, v.z};
    };
    auto get_mollifier =
          [&getV, _vertexes, _rest_vertexes] __device__(int id0, int id1, int id2, int id3) {
            auto ea0Rest = getV(_rest_vertexes[id0]);
            auto ea1Rest = getV(_rest_vertexes[id1]);
            auto eb0Rest = getV(_rest_vertexes[id2]);
            auto eb1Rest = getV(_rest_vertexes[id3]);
            T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
            auto ea0 = getV(_vertexes[id0]);
            auto ea1 = getV(_vertexes[id1]);
            auto eb0 = getV(_vertexes[id2]);
            auto eb1 = getV(_vertexes[id3]);
            return zs::make_tuple(mollifier_ee(ea0, ea1, eb0, eb1, epsX),
                                  mollifier_grad_ee(ea0, ea1, eb0, eb1, epsX));
    };
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            // ee: ++++
            auto ea0 = getV(_vertexes[MMCVIDI.x]);
            auto ea1 = getV(_vertexes[MMCVIDI.y]);
            auto eb0 = getV(_vertexes[MMCVIDI.z]);
            auto eb1 = getV(_vertexes[MMCVIDI.w]);
            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2) 
              printf("dist already smaller than xi!\n");
            auto barrierDistGrad =
                barrier_gradient(dist2 - xi2, activeGap2, Kappa);
            auto grad = eeGrad * barrierDistGrad;
            // gradient
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), grad(0, 0));
                atomicAdd(&(_gradient[MMCVIDI.x].y), grad(0, 1));
                atomicAdd(&(_gradient[MMCVIDI.x].z), grad(0, 2));
                atomicAdd(&(_gradient[MMCVIDI.y].x), grad(1, 0));
                atomicAdd(&(_gradient[MMCVIDI.y].y), grad(1, 1));
                atomicAdd(&(_gradient[MMCVIDI.y].z), grad(1, 2));
                atomicAdd(&(_gradient[MMCVIDI.z].x), grad(2, 0));
                atomicAdd(&(_gradient[MMCVIDI.z].y), grad(2, 1));
                atomicAdd(&(_gradient[MMCVIDI.z].z), grad(2, 2));
                atomicAdd(&(_gradient[MMCVIDI.w].x), grad(3, 0));
                atomicAdd(&(_gradient[MMCVIDI.w].y), grad(3, 1));
                atomicAdd(&(_gradient[MMCVIDI.w].z), grad(3, 2));
            }
        }
        else {
            // eem: +++-
            MMCVIDI.w = -MMCVIDI.w - 1;
            auto ea0 = getV(_vertexes[MMCVIDI.x]);
            auto ea1 = getV(_vertexes[MMCVIDI.y]);
            auto eb0 = getV(_vertexes[MMCVIDI.z]);
            auto eb1 = getV(_vertexes[MMCVIDI.w]);

            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2) 
                printf("dist already smaller than xi!\n");
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, Kappa);
            auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, Kappa);
            auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, Kappa);
            auto [mollifierEE, mollifierGradEE] = get_mollifier(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            auto scale = 1;
            auto scaledMollifierGrad = scale * barrierDist2 * mollifierGradEE;
            auto scaledEEGrad = scale * mollifierEE * barrierDistGrad * eeGrad;
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), scaledMollifierGrad(0, 0) + scaledEEGrad(0, 0));
                atomicAdd(&(_gradient[MMCVIDI.x].y), scaledMollifierGrad(0, 1) + scaledEEGrad(0, 1));
                atomicAdd(&(_gradient[MMCVIDI.x].z), scaledMollifierGrad(0, 2) + scaledEEGrad(0, 2));
                atomicAdd(&(_gradient[MMCVIDI.y].x), scaledMollifierGrad(1, 0) + scaledEEGrad(1, 0));
                atomicAdd(&(_gradient[MMCVIDI.y].y), scaledMollifierGrad(1, 1) + scaledEEGrad(1, 1));
                atomicAdd(&(_gradient[MMCVIDI.y].z), scaledMollifierGrad(1, 2) + scaledEEGrad(1, 2));
                atomicAdd(&(_gradient[MMCVIDI.z].x), scaledMollifierGrad(2, 0) + scaledEEGrad(2, 0));
                atomicAdd(&(_gradient[MMCVIDI.z].y), scaledMollifierGrad(2, 1) + scaledEEGrad(2, 1));
                atomicAdd(&(_gradient[MMCVIDI.z].z), scaledMollifierGrad(2, 2) + scaledEEGrad(2, 2));
                atomicAdd(&(_gradient[MMCVIDI.w].x), scaledMollifierGrad(3, 0) + scaledEEGrad(3, 0));
                atomicAdd(&(_gradient[MMCVIDI.w].y), scaledMollifierGrad(3, 1) + scaledEEGrad(3, 1));
                atomicAdd(&(_gradient[MMCVIDI.w].z), scaledMollifierGrad(3, 2) + scaledEEGrad(3, 2));
            }
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {
                // ppm: ----
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                int ppm[4] = {MMCVIDI.x, MMCVIDI.z, MMCVIDI.y, MMCVIDI.w};
                auto ea0 = getV(_vertexes[ppm[0]]);
                auto ea1 = getV(_vertexes[ppm[1]]);
                auto eb0 = getV(_vertexes[ppm[2]]);
                auto eb1 = getV(_vertexes[ppm[3]]);

                auto ppGrad = dist_grad_pp(ea0, eb0);
                auto dist2 = dist2_pp(ea0, eb0);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDist2 = barrier(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, Kappa);
                auto [mollifierEE, mollifierGradEE] = get_mollifier(ppm[0], ppm[1], ppm[2], ppm[3]);
                using GradT = zs::vec<T, 12>;

                auto scale = 1;
                auto scaledMollifierGrad = scale * barrierDist2 * mollifierGradEE;
                auto scaledPPGrad = scale * mollifierEE * barrierDistGrad * ppGrad;
                {
                    atomicAdd(&(_gradient[ppm[0]].x), scaledMollifierGrad(0, 0) + scaledPPGrad(0, 0));
                    atomicAdd(&(_gradient[ppm[0]].y), scaledMollifierGrad(0, 1) + scaledPPGrad(0, 1));
                    atomicAdd(&(_gradient[ppm[0]].z), scaledMollifierGrad(0, 2) + scaledPPGrad(0, 2));
                    atomicAdd(&(_gradient[ppm[1]].x), scaledMollifierGrad(1, 0));
                    atomicAdd(&(_gradient[ppm[1]].y), scaledMollifierGrad(1, 1));
                    atomicAdd(&(_gradient[ppm[1]].z), scaledMollifierGrad(1, 2));
                    atomicAdd(&(_gradient[ppm[2]].x), scaledMollifierGrad(2, 0) + scaledPPGrad(1, 0));
                    atomicAdd(&(_gradient[ppm[2]].y), scaledMollifierGrad(2, 1) + scaledPPGrad(1, 1));
                    atomicAdd(&(_gradient[ppm[2]].z), scaledMollifierGrad(2, 2) + scaledPPGrad(1, 2));
                    atomicAdd(&(_gradient[ppm[3]].x), scaledMollifierGrad(3, 0));
                    atomicAdd(&(_gradient[ppm[3]].y), scaledMollifierGrad(3, 1));
                    atomicAdd(&(_gradient[ppm[3]].z), scaledMollifierGrad(3, 2));
                }
            }
            else {
                // pp: -+__
                auto x0 = getV(_vertexes[v0I]);
                auto x1 = getV(_vertexes[MMCVIDI.y]);
                auto ppGrad = dist_grad_pp(x0, x1);
                auto dist2 = dist2_pp(x0, x1);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                auto grad = ppGrad * barrierDistGrad;
                {
                    atomicAdd(&(_gradient[v0I].x), grad(0, 0));
                    atomicAdd(&(_gradient[v0I].y), grad(0, 1));
                    atomicAdd(&(_gradient[v0I].z), grad(0, 2));
                    atomicAdd(&(_gradient[MMCVIDI.y].x), grad(1, 0));
                    atomicAdd(&(_gradient[MMCVIDI.y].y), grad(1, 1));
                    atomicAdd(&(_gradient[MMCVIDI.y].z), grad(1, 2));
                }
            }

        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                // pem: --+-
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                int pem[4] = {MMCVIDI.x, MMCVIDI.w, MMCVIDI.y, MMCVIDI.z};
                auto ea0 = getV(_vertexes[pem[0]]);
                auto ea1 = getV(_vertexes[pem[1]]);
                auto eb0 = getV(_vertexes[pem[2]]);
                auto eb1 = getV(_vertexes[pem[3]]);

                auto peGrad = dist_grad_pe(ea0, eb0, eb1);
                auto dist2 = dist2_pe(ea0, eb0, eb1);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDist2 = barrier(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistGrad = barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                auto barrierDistHess = barrier_hessian(dist2 - xi2, activeGap2, Kappa);
                auto [mollifierEE, mollifierGradEE] = get_mollifier(pem[0], pem[1], pem[2], pem[3]);
                using GradT = zs::vec<T, 12>;

                auto scale = 1;
                auto scaledMollifierGrad = scale * barrierDist2 * mollifierGradEE;
                auto scaledPEGrad = scale * mollifierEE * barrierDistGrad * peGrad;
                {
                    atomicAdd(&(_gradient[pem[0]].x), scaledMollifierGrad(0, 0) + scaledPEGrad(0, 0));
                    atomicAdd(&(_gradient[pem[0]].y), scaledMollifierGrad(0, 1) + scaledPEGrad(0, 1));
                    atomicAdd(&(_gradient[pem[0]].z), scaledMollifierGrad(0, 2) + scaledPEGrad(0, 2));
                    atomicAdd(&(_gradient[pem[1]].x), scaledMollifierGrad(1, 0));
                    atomicAdd(&(_gradient[pem[1]].y), scaledMollifierGrad(1, 1));
                    atomicAdd(&(_gradient[pem[1]].z), scaledMollifierGrad(1, 2));
                    atomicAdd(&(_gradient[pem[2]].x), scaledMollifierGrad(2, 0) + scaledPEGrad(1, 0));
                    atomicAdd(&(_gradient[pem[2]].y), scaledMollifierGrad(2, 1) + scaledPEGrad(1, 1));
                    atomicAdd(&(_gradient[pem[2]].z), scaledMollifierGrad(2, 2) + scaledPEGrad(1, 2));
                    atomicAdd(&(_gradient[pem[3]].x), scaledMollifierGrad(3, 0) + scaledPEGrad(2, 0));
                    atomicAdd(&(_gradient[pem[3]].y), scaledMollifierGrad(3, 1) + scaledPEGrad(2, 1));
                    atomicAdd(&(_gradient[pem[3]].z), scaledMollifierGrad(3, 2) + scaledPEGrad(2, 2));
                }
            }
            else {
                // pe: -++_
                auto p = getV(_vertexes[v0I]);
                auto e0 = getV(_vertexes[MMCVIDI.y]);
                auto e1 = getV(_vertexes[MMCVIDI.z]);
                auto peGrad = dist_grad_pe(p, e0, e1);
                auto dist2 = dist2_pe(p, e0, e1);
                if (dist2 < xi2) {
                    printf("dist already smaller than xi!\n");
                }
                auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, Kappa);
                auto grad = peGrad * barrierDistGrad;
                {
                    atomicAdd(&(_gradient[v0I].x), grad(0, 0));
                    atomicAdd(&(_gradient[v0I].y), grad(0, 1));
                    atomicAdd(&(_gradient[v0I].z), grad(0, 2));
                    atomicAdd(&(_gradient[MMCVIDI.y].x), grad(1, 0));
                    atomicAdd(&(_gradient[MMCVIDI.y].y), grad(1, 1));
                    atomicAdd(&(_gradient[MMCVIDI.y].z), grad(1, 2));
                    atomicAdd(&(_gradient[MMCVIDI.z].x), grad(2, 0));
                    atomicAdd(&(_gradient[MMCVIDI.z].y), grad(2, 1));
                    atomicAdd(&(_gradient[MMCVIDI.z].z), grad(2, 2));
                }
            }

        }
        else {
            // pt: -+++
            auto p = getV(_vertexes[v0I]);
            auto t0 = getV(_vertexes[MMCVIDI.y]);
            auto t1 = getV(_vertexes[MMCVIDI.z]);
            auto t2 = getV(_vertexes[MMCVIDI.w]);
            auto ptGrad = dist_grad_pt(p, t0, t1, t2);
            auto dist2 = dist2_pt(p, t0, t1, t2);
            if (dist2 < xi2) {
                printf("dist already smaller than xi!\n");
            }
            auto barrierDistGrad = zs::barrier_gradient(dist2 - xi2, activeGap2, Kappa);
            auto grad = ptGrad * barrierDistGrad;
            {
                atomicAdd(&(_gradient[v0I].x), grad(0, 0));
                atomicAdd(&(_gradient[v0I].y), grad(0, 1));
                atomicAdd(&(_gradient[v0I].z), grad(0, 2));
                atomicAdd(&(_gradient[MMCVIDI.y].x), grad(1, 0));
                atomicAdd(&(_gradient[MMCVIDI.y].y), grad(1, 1));
                atomicAdd(&(_gradient[MMCVIDI.y].z), grad(1, 2));
                atomicAdd(&(_gradient[MMCVIDI.z].x), grad(2, 0));
                atomicAdd(&(_gradient[MMCVIDI.z].y), grad(2, 1));
                atomicAdd(&(_gradient[MMCVIDI.z].z), grad(2, 2));
                atomicAdd(&(_gradient[MMCVIDI.w].x), grad(3, 0));
                atomicAdd(&(_gradient[MMCVIDI.w].y), grad(3, 1));
                atomicAdd(&(_gradient[MMCVIDI.w].z), grad(3, 2));
            }
        }
    }
}

__global__
void _calKineticGradient(double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__
void _calKineticEnergy(double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx] = make_double3(deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
}

__global__
void _GroundCollisionDetect(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, uint32_t* _environment_collisionPair, uint32_t* _gpNum, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double dist = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if (dist * dist > dHat) return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__
void _computeGroundGradientAndHessian(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double3* gradient, uint32_t* _gpNum, __GEIGEN__::Matrix3x3d* H3x3, uint32_t* D1Index, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    double t = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);

    //printf("H_b   dist   g_b    is  %lf  %lf  %lf\n", H_b, dist2, g_b);

    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }

    double param = 4.0 * H_b * dist2 + 2.0 * g_b;
    if (param > 0) {
        __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);
        __GEIGEN__::Matrix3x3d Hpg = __GEIGEN__::__S_Mat_multiply(nn, Kappa * param);

        int pidx = atomicAdd(_gpNum, 1);
        H3x3[pidx] = Hpg;
        D1Index[pidx] = gidx;

    }
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__
void _computeGroundGradient(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double3* gradient, uint32_t* _gpNum, __GEIGEN__::Matrix3x3d* H3x3, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    double t = dist2 - dHat;
    double g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    //double H_b = (std::log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }
}

__global__
void _computeGroundCloseVal(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double dTol, uint32_t* _closeConstraintID, double* _closeConstraintVal, uint32_t* _close_gpNum, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx] = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__
void _checkGroundCloseVal(const double3* vertexes, const double* g_offset, const double3* g_normal, int* _isChange, uint32_t* _closeConstraintID, double* _closeConstraintVal, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _closeConstraintID[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    if (dist2 < _closeConstraintVal[gidx]) {
        *_isChange = 1;
    }
}

__global__
void _reduct_MGroundDist(const double3* vertexes, const double* g_offset, const double3* g_normal, uint32_t* _environment_collisionPair, double2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double tempv = dist * dist;
    double2 temp = make_double2(1.0 / tempv, tempv);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp.x, i);
        double tempMax = __shfl_down(temp.y, i);
        temp.x = __m_max(temp.x, tempMin);
        temp.y = __m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp.x, i);
            double tempMax = __shfl_down(temp.y, i);
            temp.x = __m_max(temp.x, tempMin);
            temp.y = __m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}

__global__
void _computeSelfCloseVal(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double dTol, uint32_t* _closeConstraintID, double* _closeConstraintVal, uint32_t* _close_gpNum, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx] = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}


__global__
void _checkGroundIntersection(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, int* _isIntersect, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    //printf("%f  %f\n", *g_offset, dist);
    if (dist < 0)
        *_isIntersect = -1;
}


__global__
void _computeGroundEnergy_Reduction(double* squeue, const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double dHat, double Kappa, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;
    double temp = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__
void _reduct_min_groundTimeStep_to_double(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, const double3* moveDir, double* minStepSizes, double slackness, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    int svI = surfVertIds[idx];
    double temp = 1.0;
    double3 normal = *g_normal;
    double coef = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
    if (coef > 0.0) {
        double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;//normal
        temp = coef / (dist * slackness);
        //printf("%f\n", temp);
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    __syncthreads();*/
    //printf("%f\n", temp);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp, i);
        temp = __m_max(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp, i);
            temp = __m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }

}

__global__
void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes, const uint4* tetrahedra, const double3* moveDir, double* minStepSizes, double slackness, double errorRate, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    double ratio = 1 - slackness;

    double temp = 1.0 / _computeInjectiveStepSize_3d(vertexes, moveDir, tetrahedra[idx].x, tetrahedra[idx].y, tetrahedra[idx].z, tetrahedra[idx].w, ratio, errorRate);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp, i);
        temp = __m_max(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp, i);
            temp = __m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }

}

__global__
void _reduct_min_selfTimeStep_to_double(const double3* vertexes, const int4* _ccd_collitionPairs, const double3* moveDir, double* minStepSizes, double slackness, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    double temp = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collitionPairs[idx];

    if (MMCVIDI.x < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;

        double temp1 = point_triangle_ccd(vertexes[MMCVIDI.x],
            vertexes[MMCVIDI.y],
            vertexes[MMCVIDI.z],
            vertexes[MMCVIDI.w],
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);

        //double temp2 = doCCDVF(vertexes[MMCVIDI.x],
        //    vertexes[MMCVIDI.y],
        //    vertexes[MMCVIDI.z],
        //    vertexes[MMCVIDI.w],
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
        //    __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), 1e-9, 0.2);

        temp = 1.0 / temp1;
    }
    else {
        temp = 1.0 / edge_edge_ccd(vertexes[MMCVIDI.x],
            vertexes[MMCVIDI.y],
            vertexes[MMCVIDI.z],
            vertexes[MMCVIDI.w],
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
            __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);
    }

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down(temp, i);
        temp = __m_max(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down(temp, i);
            temp = __m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
    }

}

__global__
void _reduct_max_cfl_to_double(const double3* moveDir, double* max_double_val, uint32_t* mSVI, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = __GEIGEN__::__norm(moveDir[mSVI[idx]]);


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMax = __shfl_down(temp, i);
        temp = __m_max(temp, tempMax);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMax = __shfl_down(temp, i);
            temp = __m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        max_double_val[blockIdx.x] = temp;
    }

}

__global__
void _reduct_double3Sqn_to_double(const double3* A, double* D, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = __GEIGEN__::__squaredNorm(A[idx]);


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        //double tempMax = __shfl_down(temp, i);
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        D[blockIdx.x] = temp;
    }

}

__global__
void _reduct_double3Dot_to_double(const double3* A, const double3* B, double* D, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = __GEIGEN__::__v_vec_dot(A[idx], B[idx]);


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        //double tempMax = __shfl_down(temp, i);
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        D[blockIdx.x] = temp;
    }

}


__global__
void _getKineticEnergy_Reduction_3D(double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx])) * _masses[idx] * 0.5;

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        _energy[blockIdx.x] = temp;
    }
}

__global__
void _getFEMEnergy_Reduction_3D(double* squeue, const double3* vertexes, const uint4* tetrahedras, const __GEIGEN__::Matrix3x3d* DmInverses, const double* volume, int tetrahedraNum, double lenRate, double volRate) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = tetrahedraNum;
    if (idx >= numbers) return;

#ifdef USE_SNK
    double temp = __cal_StabbleNHK_energy_3D(vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate, volRate);
#else
    double temp = __cal_ARAP_energy_3D(vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate);
#endif
    
    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__
void _getRestStableNHKEnergy_Reduction_3D(double* squeue, const double* volume, int tetrahedraNum, double lenRate, double volRate) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = tetrahedraNum;
    if (idx >= numbers) return;

    double temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(4.0)))* volume[idx];

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__
void _getBarrierEnergy_Reduction_3D(double* squeue, const double3* vertexes, const double3* rest_vertexes, int4* _collisionPair, double _Kappa, double _dHat, int cpNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = cpNum;
    if (idx >= numbers) return;

    double temp = __cal_Barrier_energy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__
void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = vertexNum;
    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    double temp = __GEIGEN__::__v_vec_dot(b[idx], dx[idx]);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__
void __add_reduction(double* mem, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = mem[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down(temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down(temp, i);
        }
    }
    if (threadIdx.x == 0) {
        mem[blockIdx.x] = temp;
    }
}

__global__
void _stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir, int* bType, double alpha, bool moveBoundary, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (bType[idx] == 0 || moveBoundary) {
        _vertexes[idx] = __GEIGEN__::__minus(_vertexesTemp[idx], __GEIGEN__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

__global__
void _updateVelocities(double3* _vertexes, double3* _o_vertexes, double3* _velocities, int* btype, double ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (btype[idx] == 0) {
        _velocities[idx] = __GEIGEN__::__s_vec_multiply(__GEIGEN__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__
void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    if ((_btype[idx]) == -1|| (_btype[idx]) == 1) {
        _vertexes[idx] = __GEIGEN__::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__
void _updateBoundaryMoveDir(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, double PI, double alpha, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double massSum = 0;
    double angleX = PI / 2.5 * ipc_dt * alpha;
    __GEIGEN__::Matrix3x3d rotationL, rotationR;
    __GEIGEN__::__set_Mat_val(rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    __GEIGEN__::__set_Mat_val(rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    _moveDir[idx] = make_double3(0, 0, 0);
    if ((_btype[idx]) > 0) {
        _moveDir[idx] = __GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    }
    if ((_btype[idx]) < 0) {
        _moveDir[idx] = __GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
    }
}

__global__
void _computeXTilta(double3* _velocities, double3* _o_vertexes, double3* _xTilta, double ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double3 gravityDtSq = __GEIGEN__::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);//Vector3d(0, gravity, 0) * IPC_dt * IPC_dt;

    _xTilta[idx] = __GEIGEN__::__add(_o_vertexes[idx], __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(_velocities[idx], ipc_dt), gravityDtSq));//(mesh.V_prev[vI] + (mesh.velocities[vI] * IPC_dt + gravityDtSq));
}

__global__
void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    _faces[idx].x = sortIndex[_faces[idx].x];
    _faces[idx].y = sortIndex[_faces[idx].y];
    _faces[idx].z = sortIndex[_faces[idx].z];
    //printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__
void _updateEdges(uint32_t* sortIndex, uint2* _edges, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    _edges[idx].x = sortIndex[_edges[idx].x];
    _edges[idx].y = sortIndex[_edges[idx].y];
}

__global__
void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    _sVerts[idx] = sortIndex[_sVerts[idx]];
}

__global__
void _edgeTriIntersectionQuery(const double3* _vertexes, const uint2* _edges, const uint3* _faces, const AABB* _edge_bvs, const Node* _edge_nodes, int* _isIntesect, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    uint3 face = _faces[idx];
    //idx = idx + number - 1;


    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    //uint32_t self_eid = _edge_nodes[idx].element_idx;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_edge_bvs[0].upper, _edge_bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = sqrt(dHat);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx = _edge_nodes[node_id].right_idx;

        if (_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y)) {
                    if (segTriIntersect(_vertexes[_edges[obj_idx].x], _vertexes[_edges[obj_idx].y], _vertexes[face.x], _vertexes[face.y], _vertexes[face.z])) {
                        atomicAdd(_isIntesect, -1);
                        return;
                    }
                }

            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y)) {
                    if (segTriIntersect(_vertexes[_edges[obj_idx].x], _vertexes[_edges[obj_idx].y], _vertexes[face.x], _vertexes[face.y], _vertexes[face.z])) {
                        atomicAdd(_isIntesect, -1);
                        return;
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


/// <summary>
///  host code
/// </summary>
void GIPC::FREE_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaFree(_MatIndex));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs));
    CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
    CUDA_SAFE_CALL(cudaFree(_cpNum));
    CUDA_SAFE_CALL(cudaFree(_close_cpNum));
    CUDA_SAFE_CALL(cudaFree(_close_gpNum));
    CUDA_SAFE_CALL(cudaFree(_environment_collisionPair));
    CUDA_SAFE_CALL(cudaFree(_gpNum));
    //CUDA_SAFE_CALL(cudaFree(_moveDir));
    CUDA_SAFE_CALL(cudaFree(_groundNormal));
    CUDA_SAFE_CALL(cudaFree(_groundOffset));

    CUDA_SAFE_CALL(cudaFree(_faces));
    CUDA_SAFE_CALL(cudaFree(_edges));
    CUDA_SAFE_CALL(cudaFree(_surfVerts));

    pcg_data.FREE_DEVICE_MEM();

    bvh_e.FREE_DEVICE_MEM();
    bvh_f.FREE_DEVICE_MEM();
    BH.FREE_DEVICE_MEM();
}

void GIPC::MALLOC_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex, MAX_COLLITION_PAIRS_NUM * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs, MAX_COLLITION_PAIRS_NUM * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs, MAX_CCD_COLLITION_PAIRS_NUM * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_environment_collisionPair, surf_vertexNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_moveDir, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_cpNum, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_gpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundNormal, 5 * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundOffset, 5 * sizeof(double)));
    double h_offset[5] = { -1, -1, 1, -1, 1 };
    double3 H_normal[5];// = { make_double3(0, 1, 0);
    H_normal[0] = make_double3(0, 1, 0);
    H_normal[1] = make_double3(1, 0, 0);
    H_normal[2] = make_double3(-1, 0, 0);
    H_normal[3] = make_double3(0, 0, 1);
    H_normal[4] = make_double3(0, 0, -1);
    CUDA_SAFE_CALL(cudaMemcpy(_groundOffset, &h_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_groundNormal, &H_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMalloc((void**)&_faces, surface_Num * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_edges, edge_Num * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_surfVerts, surf_vertexNum * sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_cpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_gpNum, sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

    pcg_data.Malloc_DEVICE_MEM(vertexNum, tetrahedraNum);

}

template <typename TileVecT, int codim = 3>
zs::Vector<bv_t> retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, 
                          const TileVecT &vtemp,
                          const tiles_t &eles,
                          zs::wrapv<codim>, float thickness,
                          const zs::SmallString &xTag) {
  using namespace zs;
  static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
  constexpr auto space = execspace_e::cuda;
  Vector<bv_t> ret{eles.get_allocator(), eles.size()};
  pol(zs::range(eles.size()), [eles = proxy<space>({}, eles),
                               bvs = proxy<space>(ret),
                               vtemp = proxy<space>({}, vtemp),
                               codim_v = wrapv<codim>{}, xTag,
                               thickness] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = vtemp.template pack<3>(xTag, inds[0]);
    bv_t bv{x0, x0};
    for (int d = 1; d != dim; ++d)
      merge(bv, vtemp.template pack<3>(xTag, inds[d]));
    bv._min -= thickness / 2;
    bv._max += thickness / 2;
    bvs[ei] = bv;
  });
  return ret;
}

template <typename TileVecT0, typename TileVecT1, int codim = 3>
zs::Vector<bv_t> retrieve_bounding_volumes(
    zs::CudaExecutionPolicy &pol, const TileVecT0 &verts,
    const tiles_t &eles, const TileVecT1 &vtemp,
    zs::wrapv<codim>, float stepSize, float thickness,
    const zs::SmallString &xTag, const zs::SmallString &dirTag) {
  using namespace zs;
  static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
  constexpr auto space = execspace_e::cuda;
  Vector<bv_t> ret{eles.get_allocator(), eles.size()};
  pol(zs::range(eles.size()), [eles = proxy<space>({}, eles),
                               bvs = proxy<space>(ret),
                               verts = proxy<space>({}, verts),
                               vtemp = proxy<space>({}, vtemp),
                               codim_v = wrapv<codim>{}, xTag, dirTag, stepSize,
                               thickness] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = verts.template pack<3>(xTag, inds[0]);
    auto dir0 = vtemp.template pack<3>(dirTag, inds[0]);
    auto [mi, ma] = get_bounding_box(x0, x0 + stepSize * dir0);
    bv_t bv{mi, ma};
    for (int d = 1; d != dim; ++d) {
      auto x = verts.template pack<3>(xTag, inds[d]);
      auto dir = vtemp.template pack<3>(dirTag, inds[d]);
      auto [mi, ma] = get_bounding_box(x, x + stepSize * dir);
      merge(bv, mi);
      merge(bv, ma);
    }
    bv._min -= thickness / 2;
    bv._max += thickness / 2;
    bvs[ei] = bv;
  });
  return ret;
}

void GIPC::initBVH() {
    bvh_e.init(_vertexes, _rest_vertexes, _edges, _collisonPairs, _ccd_collisonPairs, _cpNum, _MatIndex, edge_Num, surf_vertexNum);
    bvh_f.init(_vertexes, _faces, _surfVerts, _collisonPairs, _ccd_collisonPairs, _cpNum, _MatIndex, surface_Num, surf_vertexNum);
}

void GIPC::init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer) {
    SceneSize = bvh_f.scene;
    bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(SceneSize.upper, SceneSize.lower));//(maxConer - minConer).squaredNorm();
    dTol = 1e-18 * bboxDiagSize2;
    meanMass = m_meanMass;
    meanVolumn = m_meanVolumn;
    dHat = /*22.5e-8*/1e-6 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(maxConer, minConer));
    BH.MALLOC_DEVICE_MEM_O(tetrahedraNum, surf_vertexNum, surface_Num, edge_Num);
    h_cpNum_last[0] = 0;
    h_cpNum_last[1] = 0;
    h_cpNum_last[2] = 0;
    h_cpNum_last[3] = 0;
    h_cpNum_last[4] = 0;
}

GIPC::~GIPC() {
    FREE_DEVICE_MEM();
}


void GIPC::GroundCollisionDetect() {
    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _GroundCollisionDetect << <blockNum, threadNum >> > (_vertexes, _surfVerts, _groundOffset, _groundNormal, _environment_collisionPair, _gpNum, dHat, numbers);
}

void GIPC::computeGroundGradientAndHessian(double3* _gradient) {
#ifndef USE_FRICTION  
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = h_gpNum;
    if (numbers < 1) {
        CUDA_SAFE_CALL(cudaMemcpy(&BH.DNum, _gpNum, sizeof(int), cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundGradientAndHessian << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _gradient, _gpNum, BH.H3x3, BH.D1Index, dHat, Kappa, numbers);
    CUDA_SAFE_CALL(cudaMemcpy(&BH.DNum, _gpNum, sizeof(int), cudaMemcpyDeviceToHost));
}

void GIPC::computeCloseGroundVal() {
    int numbers = h_gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundCloseVal << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _environment_collisionPair, dTol, _closeConstraintID, _closeConstraintVal, _close_gpNum, numbers);

}

bool GIPC::checkCloseGroundVal() {
    int numbers = h_close_gpNum;
    if (numbers < 1) return false;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    int* _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkGroundCloseVal << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxGroundDist() {
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = h_gpNum;
    if (numbers < 1)return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist << <blockNum, threadNum, sharedMsize >> > (_vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2 << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

void GIPC::computeGroundGradient(double3* _gradient) {
    int numbers = h_gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundGradient << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _gradient, _gpNum, BH.H3x3, dHat, Kappa, numbers);
}

double GIPC::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers) {
    //slackness = 0.9;
    //int numbers = h_cpNum[0];
    if (numbers < 1) return 1;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (_vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::cfl_largestSpeed(double* mqueue) {
    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, mqueue, _surfVerts, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}

double reduction2Kappa(int type, const double3* A, const double3* B, double* _queue, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double)));*/
    if (type == 0) {
        //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
        _reduct_double3Dot_to_double << <blockNum, threadNum, sharedMsize >> > (A, B, _queue, numbers);
    }
    else if (type == 1) {
        _reduct_double3Sqn_to_double << <blockNum, threadNum, sharedMsize >> > (A, _queue, numbers);
    }
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __add_reduction << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}

double GIPC::ground_largestFeasibleStepSize(double slackness, double* mqueue) {

    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));

    //if (h_cpNum[0] > 0) {
    //    double3* mvd = new double3[vertexNum];
    //    cudaMemcpy(mvd, _moveDir, sizeof(double3) * vertexNum, cudaMemcpyDeviceToHost);
    //    for (int i = 0;i < vertexNum;i++) {
    //        printf("%f  %f  %f\n", mvd[i].x, mvd[i].y, mvd[i].z);
    //    }
    //    delete[] mvd;
    //}
    _reduct_min_groundTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (_vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir, mqueue, slackness, numbers);


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets) {

    int numbers = tetrahedraNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (_vertexes, tets, _moveDir, mqueue, slackness, errorRate, numbers);


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("Injective Time step:   %f\n", 1.0 / minValue);
    //if (1.0 / minValue < 1) {
    //    system("pause");
    //}
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

void GIPC::buildCP() {

    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //bvh_f.Construct();
    bvh_f.SelfCollitionDetect(dHat);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //bvh_e.Construct();
    bvh_e.SelfCollitionDetect(dHat);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    GroundCollisionDetect();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&h_cpNum, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&h_gpNum, _gpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    /*CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));*/
}

void GIPC::buildFullCP(const double& alpha) {

    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));

    bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha);
    bvh_e.SelfCollitionFullDetect(dHat, _moveDir, alpha);

    CUDA_SAFE_CALL(cudaMemcpy(&h_ccd_cpNum, _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void GIPC::retrieveSurfaces() {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = zs::cuda_exec();
    surfaces = tiles_t{{{"inds", 3}}, surface_Num, memsrc_e::device, 0};
    cudaPol(range(surface_Num), [surfaces = proxy<space>({}, surfaces), _faces = this->_faces]__device__(int no) mutable {
        surfaces("inds", 0, no) = reinterpret_bits<float>(_faces[no].x);
        surfaces("inds", 1, no) = reinterpret_bits<float>(_faces[no].y);
        surfaces("inds", 2, no) = reinterpret_bits<float>(_faces[no].z);
    });

    surfEdges = tiles_t{{{"inds", 2}}, edge_Num, memsrc_e::device, 0};
    cudaPol(range(edge_Num), [surfEdges = proxy<space>({}, surfEdges), _edges = this->_edges]__device__(int no) mutable {
        surfEdges("inds", 0, no) = reinterpret_bits<float>(_edges[no].x);
        surfEdges("inds", 1, no) = reinterpret_bits<float>(_edges[no].y);
    });

    surfVerts = tiles_t{{{"inds", 1}}, surf_vertexNum, memsrc_e::device, 0};
    cudaPol(range(surf_vertexNum), [surfVerts = proxy<space>({}, surfVerts), _surfVerts = this->_surfVerts]__device__(int no) mutable {
        surfVerts("inds", no) = reinterpret_bits<float>(_surfVerts[no]);
    });
}

void GIPC::retrieveDirections() {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = zs::cuda_exec();
    cudaPol(range(vertexNum), [vtemp = proxy<space>({}, *p_vtemp), dirs = this->_moveDir]__device__(int no) mutable {
        vtemp("dir", 0, no) = -dirs[no].x;
        vtemp("dir", 1, no) = -dirs[no].y;
        vtemp("dir", 2, no) = -dirs[no].z;
    });
}
void GIPC::retrievePositions() {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto cudaPol = zs::cuda_exec();
    cudaPol(range(vertexNum), [vtemp = proxy<space>({}, *p_vtemp), x = this->_vertexes]__device__(int no) mutable {
        vtemp("xn", 0, no) = x[no].x;
        vtemp("xn", 1, no) = x[no].y;
        vtemp("xn", 2, no) = x[no].z;
    });
}

void GIPC::buildBVH() {
    auto cudaPol = zs::cuda_exec();
    retrievePositions();
    auto triBvs = retrieve_bounding_volumes(cudaPol, *p_vtemp, surfaces, zs::wrapv<3>{}, 0, "xn");
    triBvh.build(cudaPol, triBvs);
    auto edgeBvs = retrieve_bounding_volumes(cudaPol, *p_vtemp, surfEdges, zs::wrapv<2>{}, 0, "xn");
    edgeBvh.build(cudaPol, edgeBvs);
    bvh_f.Construct();
    bvh_e.Construct();
}

AABB* GIPC::calcuMaxSceneSize() {
    return bvh_f.getSceneSize();
}

void GIPC::buildBVH_FULLCCD(const double& alpha) {
    auto cudaPol = zs::cuda_exec();
    retrievePositions();
    retrieveDirections();
    auto triBvs = retrieve_bounding_volumes(cudaPol, *p_vtemp, surfaces, *p_vtemp, zs::wrapv<3>{}, alpha, 0, "xn", "dir");
    triBvh.build(cudaPol, triBvs);
    auto edgeBvs = retrieve_bounding_volumes(cudaPol, *p_vtemp, surfEdges, *p_vtemp, zs::wrapv<2>{}, alpha, 0, "xn", "dir");
    bvh_f.ConstructFullCCD(_moveDir, alpha);
    bvh_e.ConstructFullCCD(_moveDir, alpha);
}

void GIPC::calBarrierHessian() {
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    int numbers = h_cpNum[0];
    if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _calBarrierHessian << <blockNum, threadNum >> > (_vertexes, _rest_vertexes, _collisonPairs, BH.H12x12, BH.H9x9, BH.H6x6, BH.D4Index, BH.D3Index, BH.D2Index, _cpNum, _MatIndex, dHat, Kappa, numbers);
}


void GIPC::computeSelfCloseVal() {
    int numbers = h_cpNum[0];
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _calSelfCloseVal << <blockNum, threadNum >> > (_vertexes, _collisonPairs, _closeMConstraintID, _closeMConstraintVal,
        _close_cpNum, dTol, numbers);
}

bool GIPC::checkSelfCloseVal() {
    int numbers = h_close_cpNum;
    if (numbers < 1) return false;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    int* _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal << <blockNum, threadNum >> > (_vertexes, _isChange, _closeMConstraintID, _closeMConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxSelfDist() {
    int numbers = h_cpNum[0];
    if (numbers < 1)return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist << <blockNum, threadNum, sharedMsize >> > (_vertexes, _collisonPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2 << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

void GIPC::calBarrierGradient(double3* _gradient) {
    int numbers = h_cpNum[0];
    if (numbers < 1)return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calBarrierGradient << <blockNum, threadNum >> > (_vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, Kappa, numbers);
}


void calKineticGradient(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient << <blockNum, threadNum >> > (_vertexes, _xTilta, _gradient, _masses, numbers);
}

void calKineticEnergy(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient << <blockNum, threadNum >> > (_vertexes, _xTilta, _gradient, _masses, numbers);
}

void calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    __GEIGEN__::Matrix12x12d* Hessians, const uint32_t& offset, const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double IPC_dt) {
    int numbers = tetrahedraNum;
    if (numbers < 1) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient_hessian << <blockNum, threadNum >> > (DmInverses, vertexes, tetrahedras,
        Hessians, offset, volume, gradient, tetrahedraNum, lenRate, volRate, IPC_dt);
}

void calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate) {
    int numbers = tetrahedraNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient << <blockNum, threadNum >> > (DmInverses, vertexes, tetrahedras,
        volume, gradient, tetrahedraNum, lenRate, volRate);
}


double calcMinMovement(const double3* _moveDir, double* _queue, const int& number) {

    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir, int* bType, double alpha, bool moveBoundary, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward << <blockNum, threadNum >> > (_vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}


void updateSurfaces(uint32_t* sortIndex, uint3* _faces, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfaces << <blockNum, threadNum >> > (sortIndex, _faces, numbers);
}

void updateSurfaceEdges(uint32_t* sortIndex, uint2* _edges, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateEdges << <blockNum, threadNum >> > (sortIndex, _edges, numbers);
}

void updateSurfaceVerts(uint32_t* sortIndex, uint32_t* _sVerts, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfVerts << <blockNum, threadNum >> > (sortIndex, _sVerts, numbers);
}

void calcTetMChash(uint64_t* _MChash, const double3* _vertexes, uint4* tets, const const AABB* _MaxBv, const uint32_t* sortMapVertIndex, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcTetMChash << <blockNum, threadNum >> > (_MChash, _vertexes, tets, _MaxBv, sortMapVertIndex, number);
}

void updateVertexes(double3* o_vertexes, const double3* _vertexes, double* tempM, const double* mass, __GEIGEN__::Matrix3x3d* tempCons, int* tempBtype, const __GEIGEN__::Matrix3x3d* cons, const int* bType, const uint32_t* sortIndex, uint32_t* sortMapIndex, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _updateVertexes << <blockNum, threadNum >> > (o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortIndex, sortMapIndex, numbers);
}

void updateTetrahedras(uint4* o_tetrahedras, uint4* tetrahedras, double* tempV, const double* volum, __GEIGEN__::Matrix3x3d* tempDmInverse, const __GEIGEN__::Matrix3x3d* dmInverse, const uint32_t* sortTetIndex, const uint32_t* sortMapVertIndex, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _updateTetrahedras << <blockNum, threadNum >> > (o_tetrahedras, tetrahedras, tempV, volum, tempDmInverse, dmInverse, sortTetIndex, sortMapVertIndex, number);
}

void calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcVertMChash << <blockNum, threadNum >> > (_MChash, _vertexes, _MaxBv, number);
}

void sortGeometry(device_TetraData& TetMesh, const AABB* _MaxBv, const int& vertex_num, const int& tetradedra_num) {
    calcVertMChash(TetMesh.MChash, TetMesh.vertexes, _MaxBv, vertex_num);
    // sortIndex: ordered indices -> original indices
    // sortMapVertIndex: original indices -> ordered indices
    thrust::sequence(thrust::device_ptr<uint32_t>(TetMesh.sortIndex), thrust::device_ptr<uint32_t>(TetMesh.sortIndex) + vertex_num);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(TetMesh.MChash), thrust::device_ptr<uint64_t>(TetMesh.MChash) + vertex_num, thrust::device_ptr<uint32_t>(TetMesh.sortIndex));
    updateVertexes(TetMesh.o_vertexes, TetMesh.vertexes, TetMesh.tempDouble, TetMesh.masses, TetMesh.tempMat3x3, TetMesh.tempBoundaryType, TetMesh.Constraints, TetMesh.BoundaryType, TetMesh.sortIndex, TetMesh.sortMapVertIndex, vertex_num);
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.vertexes, TetMesh.o_vertexes, vertex_num * sizeof(double3), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.masses, TetMesh.tempDouble, vertex_num * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.Constraints, TetMesh.tempMat3x3, vertex_num * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.BoundaryType, TetMesh.tempBoundaryType, vertex_num * sizeof(int), cudaMemcpyDeviceToDevice));

    calcTetMChash(TetMesh.MChash, TetMesh.vertexes, TetMesh.tetrahedras, _MaxBv, TetMesh.sortMapVertIndex, tetradedra_num);
    thrust::sequence(thrust::device_ptr<uint32_t>(TetMesh.sortIndex), thrust::device_ptr<uint32_t>(TetMesh.sortIndex) + tetradedra_num);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(TetMesh.MChash), thrust::device_ptr<uint64_t>(TetMesh.MChash) + tetradedra_num, thrust::device_ptr<uint32_t>(TetMesh.sortIndex));
    updateTetrahedras(TetMesh.tempTetrahedras, TetMesh.tetrahedras, TetMesh.tempDouble, TetMesh.volum, TetMesh.tempMat3x3, TetMesh.DmInverses, TetMesh.sortIndex, TetMesh.sortMapVertIndex, tetradedra_num);
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.tetrahedras, TetMesh.tempTetrahedras, tetradedra_num * sizeof(uint4), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.volum, TetMesh.tempDouble, tetradedra_num * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.DmInverses, TetMesh.tempMat3x3, tetradedra_num * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyDeviceToDevice));
}

////////////////////////TO DO LATER/////////////////////////////////////////






void compute_H_b(double d, double dHat, double& H)
{
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void GIPC::suggestKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(bvh_f.scene.upper, bvh_f.scene.lower));
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    if (meanMass == 0.0) {
        kappa = 1e11 / (4.0e-16 * bboxDiagSize2 * H_b);
    }
    else {
        kappa = 1e11 * meanMass / (4.0e-16 * bboxDiagSize2 * H_b);
    }
}

void GIPC::upperBoundKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(bvh_f.scene.upper, bvh_f.scene.lower));//(maxConer - minConer).squaredNorm();
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    double kappaMax = 100 * 1e11 * meanMass / (4.0e-16 * bboxDiagSize2 * H_b);
    if (meanMass == 0.0) {
        kappaMax = 100 * 1e11 / (4.0e-16 * bboxDiagSize2 * H_b);
    }

    if (kappa > kappaMax) {
        kappa = kappaMax;
    }
}


void GIPC::initKappa(device_TetraData& TetMesh)
{
    if (h_cpNum[0] > 0) {
        double3* _GE = TetMesh.fb;
        double3* _gc = TetMesh.temp_double3Mem;
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(double3)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, vertexNum * sizeof(double3)));

        calculate_fem_gradient(TetMesh.DmInverses, TetMesh.vertexes, TetMesh.tetrahedras, TetMesh.volum,
            _GE, tetrahedraNum, FEM::lengthRate, FEM::volumeRate);

        computeGroundGradient(_gc);
        calBarrierGradient(_gc);
        double gsum = reduction2Kappa(0, _gc, _GE, pcg_data.squeue, vertexNum);
        double gsnorm = reduction2Kappa(1, _gc, _GE, pcg_data.squeue, vertexNum);
        //CUDA_SAFE_CALL(cudaFree(_gc));
        //CUDA_SAFE_CALL(cudaFree(_GE));
        double minKappa = -gsum / gsnorm;
        if (minKappa > 0.0) {
            Kappa = minKappa;
        }
        suggestKappa(Kappa);
        if (Kappa < minKappa) {
            Kappa = minKappa;
        }
        upperBoundKappa(Kappa);
    }

    //printf("Kappa ====== %f\n", Kappa);
}




void GIPC::computeGradientAndHessian(device_TetraData& TetMesh) {
    calKineticGradient(TetMesh.vertexes, TetMesh.xTilta, TetMesh.fb, TetMesh.masses, vertexNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calBarrierHessian();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calBarrierGradient(TetMesh.fb);

#ifdef USE_FRICTION
    // calFrictionGradient(TetMesh.fb, TetMesh);
    // calFrictionHessian(TetMesh);
#endif

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calculate_fem_gradient_hessian(TetMesh.DmInverses, TetMesh.vertexes, TetMesh.tetrahedras, BH.H12x12, h_cpNum[4] + h_cpNum_last[4], TetMesh.volum,
        TetMesh.fb, tetrahedraNum, FEM::lengthRate, FEM::volumeRate, IPC_dt);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(BH.D4Index + h_cpNum[4] + h_cpNum_last[4], TetMesh.tetrahedras, tetrahedraNum * sizeof(uint4), cudaMemcpyDeviceToDevice));

    computeGroundGradientAndHessian(TetMesh.fb);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}



double GIPC::Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh) {
    int numbers = tetrahedraNum;

    if (type == 0 || type == 3) {
        numbers = vertexNum;
    }
    else if (type == 2) {
        numbers = h_cpNum[0];
    }
    else if (type == 4) {
        numbers = h_gpNum;
    }
    else if (type == 5) {
        numbers = h_cpNum_last[0];
    }
    else if (type == 6) {
        numbers = h_gpNum_last;
    }
    else if (type == 7 || type == 1) {
        numbers = tetrahedraNum;
    }
    if (numbers == 0) return 0;
    double* queue = pcg_data.squeue;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch (type) {
    case 0:
        _getKineticEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (TetMesh.vertexes, TetMesh.xTilta, queue, TetMesh.masses, numbers);
        break;
    case 1:
        _getFEMEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.vertexes, TetMesh.tetrahedras, TetMesh.DmInverses, TetMesh.volum, numbers, FEM::lengthRate, FEM::volumeRate);
        break;
    case 2:
        _getBarrierEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs, Kappa, dHat, numbers);
        break;
    case 3:
        _getDeltaEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.fb, _moveDir, numbers);
        break;
    case 4:
        _computeGroundEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.vertexes, _groundOffset, _groundNormal, _environment_collisionPair, dHat, Kappa, numbers);
        break;
#if 0
    case 5:
        _getFrictionEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.vertexes, TetMesh.o_vertexes, _collisonPairs_lastH, numbers, IPC_dt, distCoord, tanBasis, lambda_lastH_scalar, dHat * IPC_dt * IPC_dt, sqrt(dHat) * IPC_dt);
        break;
    case 6:
        _getFrictionEnergy_gd_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.vertexes, TetMesh.o_vertexes, _groundNormal, _collisonPairs_lastH_gd, numbers, IPC_dt, lambda_lastH_scalar_gd, sqrt(dHat) * IPC_dt);
        break;
#endif
    case 7:
        _getRestStableNHKEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, TetMesh.volum, numbers, FEM::lengthRate, FEM::volumeRate);
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __add_reduction << <blockNum, threadNum, sharedMsize >> > (queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    double result;
    cudaMemcpy(&result, queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


double GIPC::computeEnergy(device_TetraData& TetMesh) {
    double Energy = Energy_Add_Reduction_Algorithm(0, TetMesh);

    Energy += IPC_dt * IPC_dt * Energy_Add_Reduction_Algorithm(1, TetMesh);

    Energy += Energy_Add_Reduction_Algorithm(2, TetMesh);

    Energy += Kappa * Energy_Add_Reduction_Algorithm(4, TetMesh);

#ifdef USE_FRICTION
    Energy += FEM::frictionRate * Energy_Add_Reduction_Algorithm(5, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += FEM::frictionRate * Energy_Add_Reduction_Algorithm(6, TetMesh);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    return Energy;
}

void GIPC::calculateMovingDirection(device_TetraData& TetMesh) {
    bool getLocalOpt = PCG_Process(&TetMesh, &pcg_data, BH, _moveDir, vertexNum, tetrahedraNum, IPC_dt, meanVolumn);
}


bool edgeTriIntersectionQuery(const double3* _vertexes, const uint2* _edges, const uint3* _faces, const AABB* _edge_bvs, const Node* _edge_nodes, double dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery << <blockNum, threadNum >> > (_vertexes, _edges, _faces, _edge_bvs, _edge_nodes, _isIntersect, dHat, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if (h_isITST < 0) {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh)
{
    return edgeTriIntersectionQuery(TetMesh.vertexes, bvh_e._edges, bvh_f._faces, bvh_e._bvs, bvh_e._nodes, dHat, bvh_f.face_number);
}

bool GIPC::checkGroundIntersection() {
    int numbers = h_gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if (h_isITST < 0) {
        return true;
    }
    return false;

}

bool GIPC::isIntersected(device_TetraData& TetMesh)
{
    if (checkGroundIntersection()) {
        return true;
    }

    if (checkEdgeTriIntersectionIfAny(TetMesh)) {
        return true;
    }
    return false;
}

bool GIPC::lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha)
{
    bool stopped = false;
    //buildCP();
    double lastEnergyVal = computeEnergy(TetMesh);
    double c1m = 0.0;
    double armijoParam = 1e-4;
    if (armijoParam > 0.0) {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, TetMesh);
    }

    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem, TetMesh.vertexes, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

    stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);

    bool rehash = true;

    buildBVH();
    //buildCP();
    //if (h_cpNum[0] > 0) system("pause");
    int numOfIntersect = 0;
#if 0
    while (isIntersected(TetMesh)) {
        //printf("type 0 intersection happened\n");
        alpha /= 2.0;
        numOfIntersect++;
        alpha = __m_min(cfl_alpha, alpha);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);
        buildBVH();
    }
#endif

    buildCP();
    //if (h_cpNum[0] > 0) system("pause");
    //rehash = false;

    //buildCollisionSets(mesh, sh, gd, true);
    double testingE = computeEnergy(TetMesh);

    int numOfLineSearch = 0;
    double LFStepSize = alpha;
    //double temp_c1m = c1m;
    std::cout.precision(18);
    //std::cout << "testE:    " << testingE << "      lastEnergyVal:        " << abs(lastEnergyVal- RestNHEnergy) << std::endl;
#if 1
    while ((testingE > lastEnergyVal + c1m * alpha) && alpha > 1e-3 * LFStepSize && abs(testingE - lastEnergyVal) / abs(lastEnergyVal - RestNHEnergy) > 1e-8 / IPC_dt / (1 << (numOfLineSearch + 1)) /*/ vertexNum*/) {
        //printf("testE:    %f      lastEnergyVal:        %f         clm*alpha:    %f\n", testingE, lastEnergyVal, c1m * alpha);
        std::cout << "testE:    " << testingE << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);

        buildBVH();
        buildCP();
        testingE = computeEnergy(TetMesh);
    }
#endif

    if (alpha < LFStepSize) {
        bool needRecomputeCS = false;
#if 0
        while (isIntersected(TetMesh)) {
            //printf("type 1 intersection happened\n");
            alpha /= 2.0;
            numOfIntersect++;
            alpha = __m_min(cfl_alpha, alpha);
            stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, alpha, false, vertexNum);
            buildBVH();
            needRecomputeCS = true;
        }
#endif
        if (needRecomputeCS) {
            buildCP();
        }
    }
    printf("    lineSearch time step:  %f\n", alpha);
    if (alpha >= 1.0 / (1 << (1 + numOfLineSearch)) && !numOfIntersect /*&& !numOfLineSearch*/ && abs(testingE - lastEnergyVal) / abs(lastEnergyVal - RestNHEnergy) < 1e-8 / IPC_dt / (1 << (numOfLineSearch + 1))/* / vertexNum*/ /*&& (testingE > lastEnergyVal + c1m * alpha)*/) {
        stopped = true;
    }

    return stopped;
}


void GIPC::postLineSearch(device_TetraData& TetMesh, double alpha)
{
    if (Kappa == 0.0) {
        initKappa(TetMesh);
    }
    else {

        bool updateKappa = checkCloseGroundVal();
        if (!updateKappa) {
            updateKappa = checkSelfCloseVal();
        }
        if (updateKappa) {
            Kappa *= 2.0;
            upperBoundKappa(Kappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
    //printf("------------------------------------------Kappa: %f\n", Kappa);
}

void GIPC::tempMalloc_closeConstraint() {
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintID, h_gpNum * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintVal, h_gpNum * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintID, h_cpNum[0] * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintVal, h_cpNum[0] * sizeof(double)));
}

void GIPC::tempFree_closeConstraint() {
    CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
    CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
    CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
    CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
}
int maxCOllisionPairNum = 0;
int totalCollisionPairs = 0;
int total_Cg_time = 0;
#include<vector>
#include<fstream>
std::vector<int> iterV;
int GIPC::solve_subIP(device_TetraData& TetMesh) {
    int iterCap = 10000, k = 0;

    CUDA_SAFE_CALL(cudaMemset(_moveDir, 0, vertexNum * sizeof(double3)));
    //BH.MALLOC_DEVICE_MEM_O(tetrahedraNum, h_cpNum + 1, h_gpNum);
    double totalTimeStep = 0;
    for (; k < iterCap; ++k) {
        totalCollisionPairs += h_cpNum[0];
        maxCOllisionPairNum = (maxCOllisionPairNum > h_cpNum[0]) ? maxCOllisionPairNum : h_cpNum[0];
        cudaEvent_t start, end0, end1 , end2, end3, end4;
        cudaEventCreate(&start);
        cudaEventCreate(&end0);
        cudaEventCreate(&end1);
        cudaEventCreate(&end2);
        cudaEventCreate(&end3);
        cudaEventCreate(&end4);


        BH.updateDNum(tetrahedraNum, h_cpNum + 1, h_cpNum_last + 1);

        //if (h_cpNum[2] > 0) {
        //    printf("D1 = %d\n", h_cpNum[2]);
        //    //system("pause");
        //}
        //if (h_cpNum[3] > 0) {
        //    printf("D2 = %d\n", h_cpNum[3]);
        //    //system("pause");
        //}
        //if (h_cpNum[4] > 0) {
        //    printf("D3 = %d\n", h_cpNum[4]);
        //}


        cudaEventRecord(start);
        computeGradientAndHessian(TetMesh);
        


        double distToOpt_PN = calcMinMovement(_moveDir, pcg_data.squeue, vertexNum);

        bool gradVanish = (distToOpt_PN < sqrt(1e-4 * bboxDiagSize2 * IPC_dt * IPC_dt));
        if (k && gradVanish && totalTimeStep > 1 - 1e-3) {
            break;
        }
        cudaEventRecord(end0);
        calculateMovingDirection(TetMesh);
        cudaEventRecord(end1);
        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha = __m_min(alpha, ground_largestFeasibleStepSize(slackness_a, pcg_data.squeue));
        alpha = __m_min(alpha, InjectiveStepSize(0.2, 1e-6, pcg_data.squeue, TetMesh.tetrahedras));
        alpha = __m_min(alpha, self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_cpNum[0]));
        double temp_alpha = alpha;
        double alpha_CFL = alpha;

#if 1
        buildBVH_FULLCCD(temp_alpha);
        buildFullCP(temp_alpha);
        if (h_ccd_cpNum > 0) {
            double maxSpeed = cfl_largestSpeed(pcg_data.squeue);
            alpha_CFL = sqrt(dHat) / maxSpeed * 0.5;
            alpha = __m_min(alpha, alpha_CFL);
            if (temp_alpha > 2 * alpha_CFL) {
                /*buildBVH_FULLCCD(temp_alpha);
                buildFullCP(temp_alpha);*/
                alpha = __m_min(temp_alpha, self_largestFeasibleStepSize(slackness_m, pcg_data.squeue, h_ccd_cpNum) * 0.5);
                alpha = __m_max(alpha, alpha_CFL);
            }
        }
#endif

        cudaEventRecord(end2);
        //printf("alpha:  %f\n", alpha);

        bool isStop = lineSearch(TetMesh, alpha, alpha_CFL);
        cudaEventRecord(end3);
        postLineSearch(TetMesh, alpha);
        cudaEventRecord(end4);
        //BH.FREE_DEVICE_MEM();
        //if (h_cpNum[0] > 0) return;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        float time0, time1 , time2, time3, time4;
        cudaEventElapsedTime(&time0, start, end0);
        cudaEventElapsedTime(&time1, end0, end1);
        total_Cg_time += time1;
        cudaEventElapsedTime(&time2, end1, end2);
        cudaEventElapsedTime(&time3, end2, end3);
        cudaEventElapsedTime(&time4, end3, end4);
        ////*cflTime = ptime;
        printf("time0 = %f,  time1 = %f,  time2 = %f,  time3 = %f,  time4 = %f\n", time0, time1, time2, time3, time4);
        (cudaEventDestroy(start));
        (cudaEventDestroy(end0));
        (cudaEventDestroy(end1));
        (cudaEventDestroy(end2));
        (cudaEventDestroy(end3));
        (cudaEventDestroy(end4));
        totalTimeStep += alpha;
        if (k > 10 && isStop && totalTimeStep > 1 - 1e-3) {
            break;
        }
    }
    iterV.push_back(k);
    std::ofstream outiter("iterCount.txt");
    for (int ii = 0;ii < iterV.size();ii++) {
        outiter << iterV[ii] << std::endl;
    }
    outiter.close();
    return k;
    //BH.FREE_DEVICE_MEM();
    //printf("iteration k:  %d\n", k);
}

void GIPC::updateVelocities(device_TetraData& TetMesh) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateVelocities << <blockNum, threadNum >> > (TetMesh.vertexes, TetMesh.o_vertexes, TetMesh.velocities, TetMesh.BoundaryType, IPC_dt, numbers);
}

void GIPC::updateBoundary(device_TetraData& TetMesh, double alpha) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateBoundary << <blockNum, threadNum >> > (TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateBoundaryMoveDir << <blockNum, threadNum >> > (TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, IPC_dt, FEM::PI, alpha, numbers);
}

void GIPC::computeXTilta(device_TetraData& TetMesh) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _computeXTilta << <blockNum, threadNum >> > (TetMesh.velocities, TetMesh.o_vertexes, TetMesh.xTilta, IPC_dt, numbers);
}

void GIPC::sortMesh(device_TetraData& TetMesh) {
    sortGeometry(TetMesh, calcuMaxSceneSize(), vertexNum, tetrahedraNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    updateSurfaces(TetMesh.sortMapVertIndex, _faces, surface_Num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    updateSurfaceEdges(TetMesh.sortMapVertIndex, _edges, edge_Num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    updateSurfaceVerts(TetMesh.sortMapVertIndex, _surfVerts, surf_vertexNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

int totalNT = 0;
double totalTime = 0;
int total_Frames = 0;
void GIPC::IPC_Solver(device_TetraData& TetMesh) {
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    double alpha = 1;
    cudaEventRecord(start);


    upperBoundKappa(Kappa);
    if (Kappa < 1e-16) {
        suggestKappa(Kappa);
    }
    initKappa(TetMesh);

#if 0
#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, h_cpNum[0] * sizeof(double2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, h_cpNum[0] * sizeof(__GEIGEN__::Matrix3x2d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, h_cpNum[0] * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
    buildFrictionSets();
#endif
#endif

    int k = 0;
    while (true) {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        totalNT += solve_subIP(TetMesh);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = __m_min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = __m_max(minMaxDist1.y, minMaxDist2.y);


        //std::cout << "minDist:  " << minDist << "       maxDist:  " << maxDist << std::endl;
        //std::cout << "dTol:  " << dTol << "       1e-6 * bboxDiagSize2:  " << 1e-6 * bboxDiagSize2 << std::endl;
        if ((h_cpNum[0] + h_gpNum) > 0) {

            if (minDist < dTol) {
                tempFree_closeConstraint();
                break;
            }
            else if (maxDist < 1e-6 * bboxDiagSize2) {
                tempFree_closeConstraint();
                break;
            }
            else {
                tempFree_closeConstraint();
            }
        }
        else {
            tempFree_closeConstraint();
            break;
        }

#if 0
#ifdef USE_FRICTION
        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
        CUDA_SAFE_CALL(cudaFree(distCoord));
        CUDA_SAFE_CALL(cudaFree(tanBasis));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
        CUDA_SAFE_CALL(cudaFree(_MatIndex_last));

        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));

        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, h_cpNum[0] * sizeof(double2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, h_cpNum[0] * sizeof(__GEIGEN__::Matrix3x2d)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, h_cpNum[0] * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
        buildFrictionSets();
#endif
#endif

    }

#if 0
#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
    CUDA_SAFE_CALL(cudaFree(distCoord));
    CUDA_SAFE_CALL(cudaFree(tanBasis));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
    CUDA_SAFE_CALL(cudaFree(_MatIndex_last));

    CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));
#endif
#endif

    updateVelocities(TetMesh);

    computeXTilta(TetMesh);
    cudaEventRecord(end0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float time0;
    cudaEventElapsedTime(&time0, start, end0);
    totalTime += time0;
    total_Frames++;
    printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);
    printf("boundary alpha: %f\n  finished a step\n", alpha);

    std::ofstream outTime("timeCost.txt");
    outTime << "time: " << totalTime / 1000.0 << std::endl;
    outTime << "total iter: " << totalNT << std::endl;
    outTime << "frames: " << total_Frames << std::endl;
    outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
    outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
    outTime << "totalCgTime: " << total_Cg_time << std::endl;
    outTime.close();

}