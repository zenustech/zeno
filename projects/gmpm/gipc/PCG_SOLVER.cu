#include "PCG_SOLVER.cuh"
#include "device_launch_parameters.h"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools.h"
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



__global__
void add_reduction(double* mem, int numbers) {
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

__global__ void PCG_add_Reduction_delta0(double* squeue, const __GEIGEN__::Matrix3x3d* P, const double3* b, const __GEIGEN__::Matrix3x3d* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    //double3 t_P = P[idx];
    double3 t_b = b[idx];
    __GEIGEN__::Matrix3x3d t_constraint = constraint[idx];
    /*__GEIGEN__::Matrix3x3d PInverse;
    __GEIGEN__::__Inverse(P[idx], PInverse);*/
    //double vx = 1 / t_P.x, vy = 1 / t_P.y, vz = 1 / t_P.z;
    double3 filter_b = __GEIGEN__::__M_v_multiply(t_constraint, t_b);

    double temp = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_M_multiply(filter_b, P[idx]), filter_b);//filter_b.x * filter_b.x * vx + filter_b.y * filter_b.y * vy + filter_b.z * filter_b.z * vz;

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

__global__ void PCG_add_Reduction_deltaN0(double* squeue, const __GEIGEN__::Matrix3x3d* P, const double3* b, double3* r, double3* c, const __GEIGEN__::Matrix3x3d* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //double3 t_P = P[idx];
    /*__GEIGEN__::Matrix3x3d PInverse;
    __GEIGEN__::__Inverse(P[idx], PInverse);*/
    double3 t_b = b[idx];
    __GEIGEN__::Matrix3x3d t_constraint = constraint[idx];
    double3 t_r = __GEIGEN__::__M_v_multiply(t_constraint, __GEIGEN__::__minus(t_b, r[idx]));
    double3 t_c = __GEIGEN__::__M_v_multiply(P[idx], t_r);//__GEIGEN__::__v_vec_multiply(t_r, make_double3(1 / t_P.x, 1 / t_P.y, 1 / t_P.z));
    t_c = __GEIGEN__::__M_v_multiply(t_constraint, t_c);
    r[idx] = t_r;
    c[idx] = t_c;

    double temp = __GEIGEN__::__v_vec_dot(t_r, t_c);

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

__global__ void PCG_add_Reduction_deltaN(double* squeue, double3* dx, const double3* c, double3* r, const double3* q, const __GEIGEN__::Matrix3x3d* P, double3* s, double alpha, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    //double3 t_P = P[idx];
    /*__GEIGEN__::Matrix3x3d PInverse;
    __GEIGEN__::__Inverse(P[idx], PInverse);*/
    double3 t_c = c[idx];
    double3 t_dx = dx[idx];
    double3 t_r = r[idx];
    double3 t_q = q[idx];
    double3 t_s = s[idx];

    dx[idx] = __GEIGEN__::__add(t_dx, __GEIGEN__::__s_vec_multiply(t_c, alpha));
    t_r = __GEIGEN__::__add(t_r, __GEIGEN__::__s_vec_multiply(t_q, -alpha));
    r[idx] = t_r;
    t_s = __GEIGEN__::__M_v_multiply(P[idx], t_r);//__GEIGEN__::__v_vec_multiply(t_r, make_double3(1 / t_P.x, 1 / t_P.y, 1 / t_P.z));
    s[idx] = t_s;

    double temp = __GEIGEN__::__v_vec_dot(t_r, t_s);

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

__global__ void PCG_add_Reduction_tempSum(double* squeue, const double3* c, double3* q, const __GEIGEN__::Matrix3x3d* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 t_c = c[idx];
    double3 t_q = q[idx];
    __GEIGEN__::Matrix3x3d t_constraint = constraint[idx];
    t_q = __GEIGEN__::__M_v_multiply(t_constraint, t_q);
    q[idx] = t_q;

    double temp = __GEIGEN__::__v_vec_dot(t_q, t_c);

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


__global__ void PCG_add_Reduction_force(double* squeue, const double3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 t_b = b[idx];

    double temp = __GEIGEN__::__norm(t_b);//__GEIGEN__::__mabs(t_b.x) + __GEIGEN__::__mabs(t_b.y) + __GEIGEN__::__mabs(t_b.z);

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
__global__ void __PCG_FinalStep_UpdateC(const __GEIGEN__::Matrix3x3d* constraints, const double3* s, double3* c, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double3 tempc = __GEIGEN__::__add(s[idx], __GEIGEN__::__s_vec_multiply(c[idx], rate));
    c[idx] = __GEIGEN__::__M_v_multiply(constraints[idx], tempc);
}

__global__ void __PCG_initDX(double3* dx, const double3* z, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double3 tz = z[idx];
    dx[idx] = make_double3(tz.x * rate, tz.y * rate, tz.z * rate);
}


__global__ void __PCG_Solve_AX12_b(const __GEIGEN__::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __GEIGEN__::Matrix12x12d H = Hessians[idx];
    __GEIGEN__::Vector12 tempC, tempQ;

    tempC.v[0] = c[D4Index[idx].x].x;
    tempC.v[1] = c[D4Index[idx].x].y;
    tempC.v[2] = c[D4Index[idx].x].z;

    tempC.v[3] = c[D4Index[idx].y].x;
    tempC.v[4] = c[D4Index[idx].y].y;
    tempC.v[5] = c[D4Index[idx].y].z;

    tempC.v[6] = c[D4Index[idx].z].x;
    tempC.v[7] = c[D4Index[idx].z].y;
    tempC.v[8] = c[D4Index[idx].z].z;

    tempC.v[9] = c[D4Index[idx].w].x;
    tempC.v[10] = c[D4Index[idx].w].y;
    tempC.v[11] = c[D4Index[idx].w].z;

    tempQ = __GEIGEN__::__M12x12_v12_multiply(H, tempC);

    atomicAdd(&(q[D4Index[idx].x].x), tempQ.v[0]);
    atomicAdd(&(q[D4Index[idx].x].y), tempQ.v[1]);
    atomicAdd(&(q[D4Index[idx].x].z), tempQ.v[2]);

    atomicAdd(&(q[D4Index[idx].y].x), tempQ.v[3]);
    atomicAdd(&(q[D4Index[idx].y].y), tempQ.v[4]);
    atomicAdd(&(q[D4Index[idx].y].z), tempQ.v[5]);

    atomicAdd(&(q[D4Index[idx].z].x), tempQ.v[6]);
    atomicAdd(&(q[D4Index[idx].z].y), tempQ.v[7]);
    atomicAdd(&(q[D4Index[idx].z].z), tempQ.v[8]);

    atomicAdd(&(q[D4Index[idx].w].x), tempQ.v[9]);
    atomicAdd(&(q[D4Index[idx].w].y), tempQ.v[10]);
    atomicAdd(&(q[D4Index[idx].w].z), tempQ.v[11]);
}

__global__ void __PCG_Solve_AX12_b1(const __GEIGEN__::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    extern __shared__ double sData[];
    __shared__ int offset;
    int Hid = idx / 144;
    int MRid = (idx % 144) / 12;
    int MCid = (idx % 144) % 12;
    
    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 12;
    sData[threadIdx.x] = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (12 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 12) / 12;
    int Num;// = 12 + BRid - GRtid;
    int startId = offset + BRid * 12 - 12;
    int landidx = (threadIdx.x - offset) % 12;
    if (BRid == 0) {
        Num = offset;
        startId = 0;
        landidx = threadIdx.x;
    }
    else if (BRid * 12 + offset > blockDim.x) {
        Num = blockDim.x - offset - BRid * 12 + 12;
    }
    else {
        Num = 12;
    }
    
    int iter = Num;
    for (int i = 1;i < 12;i = (i << 1)) {
        if (i < Num) {
            int tempNum = iter;
            iter = ((iter + 1) >> 1);
            if (landidx < iter) {
                if (threadIdx.x + iter < blockDim.x && threadIdx.x + iter < startId + tempNum)
                    sData[threadIdx.x] += sData[threadIdx.x + iter];
            }
        }
        __syncthreads();
        //__threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0 || GRtid == 0)
        atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), sData[threadIdx.x]);
}

__global__ void __PCG_Solve_AXALL_b2(const __GEIGEN__::Matrix12x12d* Hessians12, const __GEIGEN__::Matrix9x9d* Hessians9,
    const __GEIGEN__::Matrix6x6d* Hessians6, const __GEIGEN__::Matrix3x3d* Hessians3, const uint4* D4Index, const uint3* D3Index,
    const uint2* D2Index, const uint32_t* D1Index, const double3* c, double3* q, int numbers4, int numbers3, int numbers2, int numbers1,
    int offset4, int offset3, int offset2) {
    
    if (blockIdx.x < offset4) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numbers4) return;
        __shared__ int offset;
        int Hid = idx / 144;
        int MRid = (idx % 144) / 12;
        int MCid = (idx % 144) % 12;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 12;

        double rdata = Hessians12[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (12 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 12) / 12;
        int landidx = (threadIdx.x - offset) % 12;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot(bBoundary);
        mark = __brev(mark);
        unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 12; iter <<= 1) {
            double tmp = __shfl_down(rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 && blockIdx.x < offset4 + offset3) {
        int idx = (blockIdx.x - offset4) * blockDim.x + threadIdx.x;
        if (idx >= numbers3) return;
        __shared__ int offset;
        int Hid = idx / 81;
        int MRid = (idx % 81) / 9;
        int MCid = (idx % 81) % 9;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 9;

        double rdata = Hessians9[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (9 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 9) / 9;
        int landidx = (threadIdx.x - offset) % 9;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot(bBoundary); // a bit-mask 
        mark = __brev(mark);
        unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 9; iter <<= 1) {
            double tmp = __shfl_down(rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 + offset3 && blockIdx.x < offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3) * blockDim.x + threadIdx.x;
        if (idx >= numbers2) return;
        __shared__ int offset;
        int Hid = idx / 36;
        int MRid = (idx % 36) / 6;
        int MCid = (idx % 36) % 6;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 6;

        double rdata = Hessians6[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (6 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 6) / 6;
        int landidx = (threadIdx.x - offset) % 6;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot(bBoundary);
        mark = __brev(mark);
        unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 6; iter <<= 1) {
            double tmp = __shfl_down(rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3 - offset2) * blockDim.x + threadIdx.x;
        if (idx >= numbers1) return;
        __GEIGEN__::Matrix3x3d H = Hessians3[idx];
        double3 tempC, tempQ;

        tempC.x = c[D1Index[idx]].x;
        tempC.y = c[D1Index[idx]].y;
        tempC.z = c[D1Index[idx]].z;


        tempQ = __GEIGEN__::__M_v_multiply(H, tempC);

        atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
        atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
        atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
    }
}


__global__ void __PCG_Solve_AX12_b2(const __GEIGEN__::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset;
    int Hid = idx / 144;
    int MRid = (idx % 144) / 12;
    int MCid = (idx % 144) % 12;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 12;

    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (12 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 12) / 12;
    int landidx = (threadIdx.x - offset) % 12;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);
    
    unsigned int mark = __ballot(bBoundary); 
    mark = __brev(mark);
    unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    //mark = interval;
    //for (int iter = 1; iter & 0x1f; iter <<= 1) {
    //    int tmp = __shfl_down(mark, iter);
    //    mark = tmp > mark ? tmp : mark; 
    //}
    //mark = __shfl(mark, 0);
    //__syncthreads();

    for (int iter = 1; iter < 12; iter <<= 1) {
        double tmp = __shfl_down(rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

}

__global__ void __PCG_Solve_AX12_b3(const __GEIGEN__::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset0, offset1;
    __shared__ double tempB[36];

    int Hid = idx / 144;

    int HRtid = idx % 144;

    int MRid = (HRtid) / 12;
    int MCid = (HRtid) % 12;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 12;

    if (threadIdx.x == 0) {
        offset0 = (144 - HRtid);
        offset1 = (12 - GRtid);
    }
    __syncthreads();

    int HRid = (threadIdx.x - offset0 + 144) / 144;
    int Hlandidx = (threadIdx.x - offset0) % 144;
    if (HRid == 0) {
        Hlandidx = threadIdx.x;
    }

    int BRid = (threadIdx.x - offset1 + 12) / 12;
    int landidx = (threadIdx.x - offset1) % 12;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    if (HRid > 0 && Hlandidx < 12) {
        tempB[HRid * 12 + Hlandidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
    }
    else if (HRid == 0) {
        if (offset0 <= 12) {
            tempB[HRid * 12 + Hlandidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
        }
        else if (BRid == 1) {
            tempB[HRid * 12 + landidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
        }
    }

    __syncthreads();

    int readBid = landidx;
    if (offset0 > 12 && threadIdx.x < offset1)
        readBid = landidx + (12 - offset1);
    double rdata = Hessians[Hid].m[MRid][MCid] * tempB[HRid * 12 + readBid];//(*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot(bBoundary);
    mark = __brev(mark);
    unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);

    for (int iter = 1; iter < 12; iter <<= 1) {
        double tmp = __shfl_down(rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

}

__global__ void __PCG_AXALL_P(const __GEIGEN__::Matrix12x12d* Hessians12, const __GEIGEN__::Matrix9x9d* Hessians9, 
    const __GEIGEN__::Matrix6x6d* Hessians6, const __GEIGEN__::Matrix3x3d* Hessians3, 
    const uint4* D4Index, const uint3* D3Index, const uint2* D2Index, const uint32_t* D1Index, 
    __GEIGEN__::Matrix3x3d* P, int numbers4, int numbers3, int numbers2, int numbers1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers4 + numbers3 + numbers2 + numbers1) return;

    if (idx < numbers4) {
        int Hid = idx / 12;
        int qid = idx % 12;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians12[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else if (numbers4 <= idx && idx < numbers3 + numbers4) {
        idx -= numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians9[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else if (numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2) {
        idx -= numbers3 + numbers4;
        int Hid = idx / 6;
        int qid = idx % 6;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians6[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 3;
        int qid = idx % 3;
        atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians3[Hid].m[0][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians3[Hid].m[1][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians3[Hid].m[2][qid]);
    }
}

__global__ void __PCG_AX12_P(const __GEIGEN__::Matrix12x12d* Hessians, const uint4* D4Index, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 12;
    int qid = idx % 12;

    //double Hval = Hessians[Hid].m[qid][qid];
    //atomicAdd((&(P[*(&(D4Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
    int mid = (qid / 3) * 3;
    int tid = qid % 3;

    double Hval = Hessians[Hid].m[mid][mid + tid];
    atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid+1][mid + tid];
    atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid+2][mid + tid];
    atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
}


__global__ void __PCG_Solve_AX9_b(const __GEIGEN__::Matrix9x9d* Hessians, const uint3* D3Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __GEIGEN__::Matrix9x9d H = Hessians[idx];
    __GEIGEN__::Vector9 tempC, tempQ;

    tempC.v[0] = c[D3Index[idx].x].x;
    tempC.v[1] = c[D3Index[idx].x].y;
    tempC.v[2] = c[D3Index[idx].x].z;

    tempC.v[3] = c[D3Index[idx].y].x;
    tempC.v[4] = c[D3Index[idx].y].y;
    tempC.v[5] = c[D3Index[idx].y].z;

    tempC.v[6] = c[D3Index[idx].z].x;
    tempC.v[7] = c[D3Index[idx].z].y;
    tempC.v[8] = c[D3Index[idx].z].z;



    tempQ = __GEIGEN__::__M9x9_v9_multiply(H, tempC);

    atomicAdd(&(q[D3Index[idx].x].x), tempQ.v[0]);
    atomicAdd(&(q[D3Index[idx].x].y), tempQ.v[1]);
    atomicAdd(&(q[D3Index[idx].x].z), tempQ.v[2]);

    atomicAdd(&(q[D3Index[idx].y].x), tempQ.v[3]);
    atomicAdd(&(q[D3Index[idx].y].y), tempQ.v[4]);
    atomicAdd(&(q[D3Index[idx].y].z), tempQ.v[5]);

    atomicAdd(&(q[D3Index[idx].z].x), tempQ.v[6]);
    atomicAdd(&(q[D3Index[idx].z].y), tempQ.v[7]);
    atomicAdd(&(q[D3Index[idx].z].z), tempQ.v[8]);
}

__global__ void __PCG_AX9_P(const __GEIGEN__::Matrix9x9d* Hessians, const uint3* D3Index, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 9;
    int qid = idx % 9;

    //double Hval = Hessians[Hid].m[qid][qid];
    //atomicAdd((&(P[*(&(D3Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
    int mid = (qid / 3) * 3;
    int tid = qid % 3;

    double Hval = Hessians[Hid].m[mid][mid + tid];
    atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 1][mid + tid];
    atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 2][mid + tid];
    atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
}


__global__ void __PCG_Solve_AX9_b2(const __GEIGEN__::Matrix9x9d* Hessians, const uint3* D3Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    //extern __shared__ double sData[];
    __shared__ int offset;
    int Hid = idx / 81;
    int MRid = (idx % 81) / 9;
    int MCid = (idx % 81) % 9;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 9;
    //sData[threadIdx.x] = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
    //printf("landidx  %f  %d   %d   %d\n", sData[threadIdx.x], offset, 1, 1);
    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (9 - GRtid);// < 12 ? (12 - GRtid) : 0;
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 9) / 9;
    int landidx = (threadIdx.x - offset) % 9;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot(bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    //mark = interval;
    //for (int iter = 1; iter & 0x1f; iter <<= 1) {
    //    int tmp = __shfl_down(mark, iter);
    //    mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    //}
    //mark = __shfl(mark, 0);
    //__syncthreads();

    for (int iter = 1; iter < 9; iter <<= 1) {
        double tmp = __shfl_down(rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
}

__global__ void __PCG_Solve_AX6_b(const __GEIGEN__::Matrix6x6d* Hessians, const uint2* D2Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __GEIGEN__::Matrix6x6d H = Hessians[idx];
    __GEIGEN__::Vector6 tempC, tempQ;

    tempC.v[0] = c[D2Index[idx].x].x;
    tempC.v[1] = c[D2Index[idx].x].y;
    tempC.v[2] = c[D2Index[idx].x].z;

    tempC.v[3] = c[D2Index[idx].y].x;
    tempC.v[4] = c[D2Index[idx].y].y;
    tempC.v[5] = c[D2Index[idx].y].z;



    tempQ = __GEIGEN__::__M6x6_v6_multiply(H, tempC);

    atomicAdd(&(q[D2Index[idx].x].x), tempQ.v[0]);
    atomicAdd(&(q[D2Index[idx].x].y), tempQ.v[1]);
    atomicAdd(&(q[D2Index[idx].x].z), tempQ.v[2]);

    atomicAdd(&(q[D2Index[idx].y].x), tempQ.v[3]);
    atomicAdd(&(q[D2Index[idx].y].y), tempQ.v[4]);
    atomicAdd(&(q[D2Index[idx].y].z), tempQ.v[5]);
}

__global__ void __PCG_AX6_P(const __GEIGEN__::Matrix6x6d* Hessians, const uint2* D2Index, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 6;
    int qid = idx % 6;

    //double Hval = Hessians[Hid].m[qid][qid];
    //atomicAdd((&(P[*(&(D2Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
    int mid = (qid / 3) * 3;
    int tid = qid % 3;

    double Hval = Hessians[Hid].m[mid][mid + tid];
    atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 1][mid + tid];
    atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 2][mid + tid];
    atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
}


__global__ void __PCG_Solve_AX6_b2(const __GEIGEN__::Matrix6x6d* Hessians, const uint2* D2Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset;
    int Hid = idx / 36;
    int MRid = (idx % 36) / 6;
    int MCid = (idx % 36) % 6;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 6;

    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (6 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 6) / 6;
    int landidx = (threadIdx.x - offset) % 6;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot(bBoundary); 
    mark = __brev(mark);
    unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    //mark = interval;
    //for (int iter = 1; iter & 0x1f; iter <<= 1) {
    //    int tmp = __shfl_down(mark, iter);
    //    mark = tmp > mark ? tmp : mark; 
    //}
    //mark = __shfl(mark, 0);
    //__syncthreads();

    for (int iter = 1; iter < 6; iter <<= 1) {
        double tmp = __shfl_down(rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
}


__global__ void __PCG_Solve_AX3_b(const __GEIGEN__::Matrix3x3d* Hessians, const uint32_t* D1Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __GEIGEN__::Matrix3x3d H = Hessians[idx];
    double3 tempC, tempQ;

    tempC.x = c[D1Index[idx]].x;
    tempC.y = c[D1Index[idx]].y;
    tempC.z = c[D1Index[idx]].z;


    tempQ = __GEIGEN__::__M_v_multiply(H, tempC);

    atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
    atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
    atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
}

__global__ void __PCG_AX3_P(const __GEIGEN__::Matrix3x3d* Hessians, const uint32_t* D1Index, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 3;
    int qid = idx % 3;

    //double Hval = Hessians[Hid].m[qid][qid];
    //*(&(P[(D1Index[Hid])].x) + qid) += Hval;
    //P[D1Index[Hid]].m[0][qid] += Hessians[Hid].m[0][qid];
    //P[D1Index[Hid]].m[1][qid] += Hessians[Hid].m[1][qid];
    //P[D1Index[Hid]].m[2][qid] += Hessians[Hid].m[2][qid];
    atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians[Hid].m[0][qid]);
    atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians[Hid].m[1][qid]);
    atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians[Hid].m[2][qid]);
}


__global__ void __PCG_Solve_AX3_b2(const __GEIGEN__::Matrix3x3d* Hessians, const uint32_t* D1Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset;
    int Hid = idx / 9;
    int MRid = (idx % 9) / 3;
    int MCid = (idx % 9) % 3;


    int axisId = MCid % 3;
    int GRtid = idx % 3;

    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[(D1Index[Hid])].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (3 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 3) / 3;
    int landidx = (threadIdx.x - offset) % 3;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot(bBoundary);  
    mark = __brev(mark);
    unsigned int interval = __m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down(mark, iter);
        mark = tmp > mark ? tmp : mark; 
    }
    mark = __shfl(mark, 0);
    __syncthreads();

    for (int iter = 1; iter <= mark; iter <<= 1) {
        double tmp = __shfl_down(rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[(D1Index[Hid])].x) + MRid % 3), rdata);
}


__global__ void __PCG_Solve_AX_mass_b(const double* _masses, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;


    double3 tempQ = __GEIGEN__::__s_vec_multiply(c[idx], _masses[idx]);

    atomicAdd(&(q[idx].x), tempQ.x);
    atomicAdd(&(q[idx].y), tempQ.y);
    atomicAdd(&(q[idx].z), tempQ.z);
}



__global__ void __PCG_mass_P(const double* _masses, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double mass = _masses[idx];
    __GEIGEN__::__init_Mat3x3(P[idx], 0);
    P[idx].m[0][0] = mass;
    P[idx].m[1][1] = mass;
    P[idx].m[2][2] = mass;
}

__global__ void __PCG_init_P(const double* _masses, __GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    __GEIGEN__::__init_Mat3x3(P[idx], 0);
    P[idx].m[0][0] = 1;
    P[idx].m[1][1] = 1;
    P[idx].m[2][2] = 1;
}

__global__ void __PCG_inverse_P(__GEIGEN__::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    __GEIGEN__::Matrix3x3d PInverse;
    __GEIGEN__::__Inverse(P[idx], PInverse);

    P[idx] = PInverse;

}

double My_PCG_add_Reduction_Algorithm(int type, device_TetraData* mesh, PCG_Data* pcg_data, int vertexNum, double alpha = 1) {

    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch (type) {
    case 0:
        PCG_add_Reduction_force << <blockNum, threadNum, sharedMsize >> > (pcg_data->squeue, pcg_data->b, numbers);
        break;
    case 1:
        PCG_add_Reduction_delta0 << <blockNum, threadNum, sharedMsize >> > (pcg_data->squeue, pcg_data->P, pcg_data->b, mesh->Constraints, numbers);
        break;
    case 2:
        PCG_add_Reduction_deltaN0 << <blockNum, threadNum, sharedMsize >> > (pcg_data->squeue, pcg_data->P, pcg_data->b, pcg_data->r, pcg_data->c, mesh->Constraints, numbers);
        break;
    case 3:
        PCG_add_Reduction_tempSum << <blockNum, threadNum, sharedMsize >> > (pcg_data->squeue, pcg_data->c, pcg_data->q, mesh->Constraints, numbers);
        break;
    case 4:
        PCG_add_Reduction_deltaN << <blockNum, threadNum, sharedMsize >> > (pcg_data->squeue, pcg_data->dx, pcg_data->c, pcg_data->r, pcg_data->q, pcg_data->P, pcg_data->s, alpha, numbers);
        break;
    }

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        add_reduction << <blockNum, threadNum, sharedMsize >> > (pcg_data->squeue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    double result;
    cudaMemcpy(&result, pcg_data->squeue, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

void Solve_PCG_AX_B(const device_TetraData* mesh, const double3* c, double3* q, const BHessian& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Solve_AX_mass_b << <blockNum, threadNum >> > (mesh->masses, c, q, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = BH.DNum[3];
    if (numbers > 0) {
        //unsigned int sharedMsize = sizeof(double) * threadNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX12_b << <blockNum, threadNum>> > (BH.H12x12, BH.D4Index, c, q, numbers);
    }
    numbers = BH.DNum[2];
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX9_b << <blockNum, threadNum >> > (BH.H9x9, BH.D3Index, c, q, numbers);
    }
    numbers = BH.DNum[1];
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX6_b << <blockNum, threadNum >> > (BH.H6x6, BH.D2Index, c, q, numbers);
    }
    numbers = BH.DNum[0];
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX3_b << <blockNum, threadNum >> > (BH.H3x3, BH.D1Index, c, q, numbers);
    }
    
}

void Solve_PCG_AX_B2(const device_TetraData* mesh, const double3* c, double3* q, const BHessian& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Solve_AX_mass_b << <blockNum, threadNum >> > (mesh->masses, c, q, numbers);

    int offset4 = (BH.DNum[3] * 144 + threadNum - 1) / threadNum;
    int offset3 = (BH.DNum[2] * 81 + threadNum - 1) / threadNum;
    int offset2 = (BH.DNum[1] * 36 + threadNum - 1) / threadNum;
    int offset1 = (BH.DNum[0] + threadNum - 1) / threadNum;
    blockNum = offset1 + offset2 + offset3 + offset4;
    __PCG_Solve_AXALL_b2 << <blockNum, threadNum>> > (BH.H12x12, BH.H9x9, BH.H6x6, BH.H3x3, BH.D4Index, BH.D3Index, BH.D2Index, BH.D1Index, c, q, BH.DNum[3] * 144, BH.DNum[2] * 81, BH.DNum[1] * 36, BH.DNum[0], offset4, offset3, offset2);

}

void construct_P(const device_TetraData* mesh, __GEIGEN__::Matrix3x3d* P, const BHessian& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_mass_P << <blockNum, threadNum >> > (mesh->masses, P, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = BH.DNum[3] * 12;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX12_P << <blockNum, threadNum >> > (BH.H12x12, BH.D4Index, P, numbers);
    }
    numbers = BH.DNum[2] * 9;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX9_P << <blockNum, threadNum >> > (BH.H9x9, BH.D3Index, P, numbers);
    }
    numbers = BH.DNum[1] * 6;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX6_P << <blockNum, threadNum >> > (BH.H6x6, BH.D2Index, P, numbers);
    }
    numbers = BH.DNum[0] * 3;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX3_P << <blockNum, threadNum >> > (BH.H3x3, BH.D1Index, P, numbers);
    }
    blockNum = (vertNum + threadNum - 1) / threadNum;
    __PCG_inverse_P << <blockNum, threadNum >> > (P, vertNum);
    //__PCG_init_P << <blockNum, threadNum >> > (mesh->masses, P, vertNum);
}

void construct_P2(const device_TetraData* mesh, __GEIGEN__::Matrix3x3d* P, const BHessian& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_mass_P << <blockNum, threadNum >> > (mesh->masses, P, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = BH.DNum[3] * 12 + BH.DNum[2] * 9 + BH.DNum[1] * 6 + BH.DNum[0] * 3;
    blockNum = (numbers + threadNum - 1) / threadNum;

    __PCG_AXALL_P << <blockNum, threadNum >> > (BH.H12x12, BH.H9x9, BH.H6x6, BH.H3x3, BH.D4Index, BH.D3Index, BH.D2Index, BH.D1Index, P, BH.DNum[3] * 12, BH.DNum[2] * 9, BH.DNum[1] * 6, BH.DNum[0] * 3);

    blockNum = (vertNum + threadNum - 1) / threadNum;
    __PCG_inverse_P << <blockNum, threadNum >> > (P, vertNum);
    //__PCG_init_P << <blockNum, threadNum >> > (mesh->masses, P, vertNum);
}

void PCG_FinalStep_UpdateC(const device_TetraData* mesh, double3* c, const double3* s, const double& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_FinalStep_UpdateC << <blockNum, threadNum >> > (mesh->Constraints, s, c, rate, numbers);
}

void PCG_initDX(double3* dx, const double3* z, double rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_initDX << <blockNum, threadNum >> > (dx, z, rate, numbers);
}

bool PCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn) {
    construct_P2(mesh, pcg_data->P, BH, vertexNum);
    double deltaN = 0;
    double delta0 = 0;
    double deltaO = 0;
    //PCG_initDX(pcg_data->dx, pcg_data->z, 0.5, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(pcg_data->dx, 0x0, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(pcg_data->r, 0x0, vertexNum * sizeof(double3)));
    delta0 = My_PCG_add_Reduction_Algorithm(1, mesh, pcg_data, vertexNum);
    //Solve_PCG_AX_B2(mesh, pcg_data->z, pcg_data->r, BH, vertexNum);
    deltaN = My_PCG_add_Reduction_Algorithm(2, mesh, pcg_data, vertexNum);
    //std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " << deltaN << std::endl;
    double errorRate = std::min(1e-8 * 0.5 * IPC_dt / std::pow(meanVolumn, 1), 1e-4);
    //printf("cg error Rate:   %f        meanVolumn: %f\n", errorRate, meanVolumn);
    int cgCounts = 0;
    while (cgCounts<30000 && deltaN > errorRate * delta0) {
        cgCounts++;
        //std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN << "      iteration_counts:      " << cgCounts << std::endl;
        CUDA_SAFE_CALL(cudaMemset(pcg_data->q, 0, vertexNum * sizeof(double3)));
        Solve_PCG_AX_B2(mesh, pcg_data->c, pcg_data->q, BH, vertexNum);
        double tempSum = My_PCG_add_Reduction_Algorithm(3, mesh, pcg_data, vertexNum);
        double alpha = deltaN / tempSum;
        deltaO = deltaN;
        deltaN = 0;
        CUDA_SAFE_CALL(cudaMemset(pcg_data->s, 0, vertexNum * sizeof(double3)));
        deltaN = My_PCG_add_Reduction_Algorithm(4, mesh, pcg_data, vertexNum, alpha);
        double rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(mesh, pcg_data->c, pcg_data->s, rate, vertexNum);
        //cudaDeviceSynchronize();
    }
    _mvDir = pcg_data->dx;
    //CUDA_SAFE_CALL(cudaMemcpy(pcg_data->z, _mvDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));
    printf("cg counts = %d\n", cgCounts);
    return true;
}

void PCG_Data::Malloc_DEVICE_MEM(const int& vertexNum, const int& tetrahedraNum) {
    //std::cout << vertexNum << std::endl;
    //int maxNum = __m_max(vertexNum, tetrahedraNum);
    CUDA_SAFE_CALL(cudaMalloc((void**)&squeue, __m_max(vertexNum, tetrahedraNum) * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&b, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&P, vertexNum * sizeof(__GEIGEN__::Matrix3x3d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&r, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&c, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&z, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&q, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&s, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dx, vertexNum * sizeof(double3)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&tempDx, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(z, 0, vertexNum * sizeof(double3)));
}

void PCG_Data::FREE_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaFree(squeue));
    //CUDA_SAFE_CALL(cudaFree(b));
    CUDA_SAFE_CALL(cudaFree(P));
    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(c));
    CUDA_SAFE_CALL(cudaFree(z));
    CUDA_SAFE_CALL(cudaFree(q));
    CUDA_SAFE_CALL(cudaFree(s));
    CUDA_SAFE_CALL(cudaFree(dx));
    //CUDA_SAFE_CALL(cudaFree(tempDx));
}

void BHessian::updateDNum(const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums) {

    DNum[1] = cpNums[1];
    DNum[2] = cpNums[2];
    DNum[3] = tet_number + cpNums[3];

#ifdef USE_FRICTION
    DNum[1] += last_cpNums[1];
    DNum[2] += last_cpNums[2];
    DNum[3] += last_cpNums[3];
#endif
}

void BHessian::MALLOC_DEVICE_MEM_O(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number) {

    CUDA_SAFE_CALL(cudaMalloc((void**)&H12x12, 3 * (tet_number + surfvert_number + surfEdge_number) * sizeof(__GEIGEN__::Matrix12x12d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&H9x9, 5 * (surfEdge_number + surfvert_number) * sizeof(__GEIGEN__::Matrix9x9d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&H6x6, 3 * (surfvert_number + surfEdge_number) * sizeof(__GEIGEN__::Matrix6x6d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&H3x3, 2 * surfvert_number * sizeof(__GEIGEN__::Matrix3x3d)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&D1Index, 2 * surfvert_number * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&D2Index, 3 * (surfvert_number + surfEdge_number) * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&D3Index, 5 * (surfEdge_number + surfvert_number) * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&D4Index, 3 * (tet_number + surfvert_number + surfEdge_number) * sizeof(uint4)));
}

void BHessian::FREE_DEVICE_MEM() {
    CUDA_SAFE_CALL(cudaFree(H12x12));
    CUDA_SAFE_CALL(cudaFree(H9x9));
    CUDA_SAFE_CALL(cudaFree(H6x6));
    CUDA_SAFE_CALL(cudaFree(H3x3));
    CUDA_SAFE_CALL(cudaFree(D1Index));
    CUDA_SAFE_CALL(cudaFree(D2Index));
    CUDA_SAFE_CALL(cudaFree(D3Index));
    CUDA_SAFE_CALL(cudaFree(D4Index));
}
