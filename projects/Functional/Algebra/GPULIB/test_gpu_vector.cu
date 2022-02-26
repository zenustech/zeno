


#include "utilities.h"


namespace gpu {


template<typename T>
__global__ void feprojection_form_kernel(const T* x, T* Ax, size_t num)
{
    double A[3][3] = {
        {1.0, 1.0, 3.0},
        {4.0, 3.0, 0.0},
        {8.0, 0.0, 0.0}
    };

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gidx<num){
        Ax[gidx] = 0.0;
        for (size_t j = 0; j < 3; j++)
        {
            Ax[gidx] += A[gidx][j]*x[j];
        }   
    }

}
template<typename T>
void gpu_feprojection_form(const T* x, T* Ax, size_t num){
    uint numThreads, numBlocks;
    computeGridSize(num, 8, numBlocks, numThreads);
    feprojection_form_kernel<<<numBlocks, numThreads>>>(x,Ax,num);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");
}


template void gpu_feprojection_form<double>(const double* x, double* Ax, size_t num);
template void gpu_feprojection_form<float>(const float* x, float* Ax, size_t num);





// gpu::gpu_nonlinear_residual(_x.data, _r.data, _x.num);



template<typename T>
__global__ void nonlinear_residual_kernel(const T* x, T* r, size_t num)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gidx == 0){
        r[0] = exp(2.0*x[0])/2.0 - x[1];
        r[1] = x[0]*x[0] + x[1]*x[1]-1.0;
    }
}
template<typename T>
void gpu_nonlinear_residual(const T* x, T* r, size_t num){
    uint numThreads, numBlocks;
    computeGridSize(num, 8, numBlocks, numThreads);
    nonlinear_residual_kernel<<<numBlocks, numThreads>>>(x,r,num);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");
}


template void gpu_nonlinear_residual<double>(const double* x, double* r, size_t num);
template void gpu_nonlinear_residual<float>(const float* x, float* r, size_t num);




}
