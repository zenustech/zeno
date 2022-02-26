/**
 * @file ImmersedBoundaryMethod.cu
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-07
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */
#include"ImmersedBoundaryMethod.h"

namespace gpu{
__forceinline__ __device__ double ibm_phi(double r)
{
    double res=0;
    double r2 = r*r;
    if(r<=2)
        res = 0.125*(5-2*r-sqrt(-7+12*r-4*r2));
    if(r<=1)
        res = 0.125*(3-2*r+sqrt(1+4*r-4*r2));
    return res;
}

__forceinline__ __device__ double ibm_delta3(double3 x, double3 g, double h)
{

    return ibm_phi(abs(x.x-g.x))*ibm_phi(abs(x.y-g.y))*ibm_phi(abs(x.z-g.z));
}

__forceinline__ __device__ double ibm_delta3(double3 x, double3 g)
{

    return ibm_phi(abs(x.x-g.x))*ibm_phi(abs(x.y-g.y))*ibm_phi(abs(x.z-g.z));
}

/// points Spread force from solid points to grid points
/// inspired by xinxin's immersed boudnary method
__global__ void distribute_force_kernel(
    const double3* solid_forces,                        /// solid_forces
    const double4* quadrature_rules,                    /// quadrature quadrature_rules
          double3* fluid_forces,
    int num, double h, int3 dim)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx<num)
    {
        double4 quadrature_rule = quadrature_rules[gidx];
        int i = floor(quadrature_rule.x/h);
        int j = floor(quadrature_rule.y/h);
        int k = floor(quadrature_rule.z/h);
        double w = quadrature_rule.w;
        double3 f = solid_forces[gidx];
        double inv_h3 = 1.0/h/h/h;

        // printf("weights : %e, h : %f, f.x : %f, f.y : %f, f.z : %f, inv_h3 : %f. \n", 
        //                    w, h, f.x, f.y, f.z, inv_h3);

        for(int kk=k-1;kk<=k+2;kk++)
        {
            for(int jj=j-1;jj<=j+2;jj++)
            {
                for(int ii=i-1;ii<=i+2;ii++)
                {
                    if(!(kk>=0&&kk<dim.z&&jj>=0&&jj<dim.y&&ii>=0&&ii<dim.x))
                    {
                        continue;
                    }
                    int widx = ii + jj*dim.x + kk*dim.x*dim.y;
                    double3 spreadf;
                    double weight = w*inv_h3*ibm_delta3(make_double3(quadrature_rule.x/h, quadrature_rule.y/h, quadrature_rule.z/h), make_double3(ii,jj,kk));
                    spreadf.x = f.x*weight;
                    spreadf.y = f.y*weight;
                    spreadf.z = f.z*weight;
                    // it can not be used togather with "continue"
                    // __syncthreads();
                    atomicAdd(&(fluid_forces[widx].x), (spreadf.x));
                    atomicAdd(&(fluid_forces[widx].y), (spreadf.y));
                    atomicAdd(&(fluid_forces[widx].z), (spreadf.z));
                    
                    // if (fabs(fluid_forces[widx].x) > 0.01)
                    //     printf("distribute_force_kernel : %d, %lf, %lf, %lf\n", widx, fluid_forces[widx].x, fluid_forces[widx].y, fluid_forces[widx].z);
                }
            }
        }
    }
}


__global__ void interpolate_velocity_kernel(
          double3* solid_velocities, 
    const double4* quadrature_rules, 
    const double3* fluid_velocities, 
    int num, double h, int3 dim)
{
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx<num) {
        double4 pos = quadrature_rules[gidx];
        int i = floor(pos.x/h);
        int j = floor(pos.y/h);
        int k = floor(pos.z/h);
        double3 sum = make_double3(0,0,0);
        for (int kk = k - 1; kk <= k + 2; kk++) {
            for (int jj = j - 1; jj <= j + 2; jj++) {
                for (int ii = i - 1; ii <= i + 2; ii++) {
                    if (!(kk >= 0 && kk < dim.z && jj >= 0 && jj < dim.y && ii >= 0 && ii < dim.x)) {
                        continue;
                    }
                    // NOTE : which one is correct?
                    // int ridx = ii + jj * dim.x + kk * dim.x * dim.y;
                    int ridx = ii + jj * dim.x + kk * dim.x * dim.y;
                    double weight = ibm_delta3(make_double3(pos.x / h, pos.y / h, pos.z / h),
                                               make_double3(ii, jj, kk), h);
                    // BUG : __syncthreads() here is a bug.
                    // __syncthreads();
                    double3 gvalue = fluid_velocities[ridx];
                    sum.x += weight * gvalue.x;
                    sum.y += weight * gvalue.y;
                    sum.z += weight * gvalue.z;
                    // printf("interpolate_velocity_kernel : %d, %lf, %lf, %lf\n", ridx, gvalue.x, gvalue.y, gvalue.z);
                }
            }
        }
        // BUG : __syncthreads() here is a bug.
        // __syncthreads();
        solid_velocities[gidx] = sum;
    }
}


    void distribute_force_gpu(
        const double3* solid_forces,                        /// solid_forces
        const double4* quadrature_rules,                    /// quadrature quadrature_rules
              double3* fluid_forces,
        size_t num, double h, int3 dim)
    {
        std::cout << "\n\n call kernel distribute_force.\n\n" << std::endl;
        uint numThreads, numBlocks;
        computeGridSize(num, 256, numBlocks, numThreads);
        distribute_force_kernel<<<numBlocks, numThreads>>>(solid_forces, quadrature_rules, fluid_forces, num, h, dim);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    
    void interpolate_velocity_gpu(
              double3* solid_velocities, 
        const double4* quadrature_rules, 
        const double3* fluid_velocities, 
        size_t num, double h, int3 dim
    ){
        std::cout << "\n\n call function interpolate_velocity_gpu:\n\n" << std::endl;
        uint numThreads, numBlocks;
        computeGridSize(num, 256, numBlocks, numThreads);
        interpolate_velocity_kernel<<<numBlocks, numThreads>>>(solid_velocities, quadrature_rules, fluid_velocities, num, h, dim);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    // Stay the night because I need this more than I knew, more than I would like, more than you do.
    // I need this function now not in the future.
    void distribute_force(
        const double3* solid_forces,                        /// solid_forces
        const double4* quadrature_rules,                    /// quadrature quadrature_rules
              double3* fluid_forces,
        size_t num, double h, int3 dim, bool useCUDA = true)
    {
        if(useCUDA){
            cudaDeviceSynchronize();
            Timer timer("distribute_force with cuda.");

            double3* solid_forces_dev;
            double4* quadrature_rules_dev;
            double3* fluid_forces_dev;

            // Malloc memory
            checkCudaErrors(cudaMalloc((void **)&solid_forces_dev, num*sizeof(double3)));
            checkCudaErrors(cudaMalloc((void **)&quadrature_rules_dev, num*sizeof(double4)));
            checkCudaErrors(cudaMalloc((void **)&fluid_forces_dev, dim.x*dim.y*dim.z*sizeof(double3)));

            // Copy 
            checkCudaErrors(cudaMemcpy(solid_forces_dev, solid_forces, num*sizeof(double3), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(quadrature_rules_dev, quadrature_rules, num*sizeof(double4), cudaMemcpyHostToDevice));

            // Call
            distribute_force_gpu(solid_forces_dev, quadrature_rules_dev, fluid_forces_dev, num, h, dim);

            // Copy back the results
            checkCudaErrors(cudaMemcpy(fluid_forces, fluid_forces_dev, dim.x*dim.y*dim.z*sizeof(double3), cudaMemcpyDeviceToHost));
        
            // for (size_t i = 0; i < dim.x*dim.y*dim.z; i++)
            // {
            //     if (fabs(fluid_forces[i].x) > 0.01)
            //         printf("distribute_force_kernel : %d, %lf, %lf, %lf\n", i, fluid_forces[i].x, fluid_forces[i].y, fluid_forces[i].z);
            // }

            // Free
            checkCudaErrors(cudaFree(solid_forces_dev));
            checkCudaErrors(cudaFree(quadrature_rules_dev));
            checkCudaErrors(cudaFree(fluid_forces_dev));
            cudaDeviceSynchronize();
        } else {
            cudaDeviceSynchronize();
            Timer timer("distribute_force without cuda.");
            distribute_force_cpu(solid_forces, quadrature_rules, fluid_forces, num, h, dim);
            cudaDeviceSynchronize();
        }
    }


    void interpolate_velocity(
              double3* solid_velocities,                        /// solid_forces
        const double4* quadrature_rules,                    /// quadrature quadrature_rules
        const double3* fluid_velocities,
        size_t num, double h, int3 dim, bool useCUDA = true)
    {
        if (useCUDA)
        {
            cudaDeviceSynchronize();
            Timer timer("interpolate_velocity with cuda.");
        
            double3* solid_velocities_dev;
            double4* quadrature_rules_dev;
            double3* fluid_velocities_dev;

            // Malloc memory
            checkCudaErrors(cudaMalloc((void **)&solid_velocities_dev, num*sizeof(double3)));
            checkCudaErrors(cudaMalloc((void **)&quadrature_rules_dev, num*sizeof(double4)));
            checkCudaErrors(cudaMalloc((void **)&fluid_velocities_dev, dim.x*dim.y*dim.z*sizeof(double3)));
                        
            // for (size_t i = 0; i < dim.x*dim.y*dim.z; i++)
            // {
            //     printf("interpolate_velocity_cpu : %d, %lf, %lf, %lf\n", i, fluid_velocities[i].x, fluid_velocities[i].y, fluid_velocities[i].z);
            //     /* code */
            // }
        
            // Copy 
            checkCudaErrors(cudaMemcpy(fluid_velocities_dev, fluid_velocities, dim.x*dim.y*dim.z*sizeof(double3), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(quadrature_rules_dev, quadrature_rules, num*sizeof(double4), cudaMemcpyHostToDevice));
            
            // Call
            interpolate_velocity_gpu(solid_velocities_dev, quadrature_rules_dev, fluid_velocities_dev, num, h, dim);

            // Copy back the results
            checkCudaErrors(cudaMemcpy(solid_velocities, solid_velocities_dev, num*sizeof(double3), cudaMemcpyDeviceToHost));

            // Free
            checkCudaErrors(cudaFree(solid_velocities_dev));
            checkCudaErrors(cudaFree(quadrature_rules_dev));
            checkCudaErrors(cudaFree(fluid_velocities_dev));
            cudaDeviceSynchronize();
        }
        else {
            cudaDeviceSynchronize();
            Timer timer("interpolate_velocity without cuda.");
            interpolate_velocity_cpu(solid_velocities, quadrature_rules, fluid_velocities, num, h, dim);
            cudaDeviceSynchronize();
        }
    }


}