#include "utilities.h"

namespace gpu {

__device__ double det_3x3(const double* m){
    return m[0]*(m[4]*m[8]-m[7]*m[5]) - m[1]*(m[3]*m[8]-m[6]*m[5])+m[2]*(m[3]*m[7]-m[6]*m[4]);
}

__device__ void inverse_3x3(const double* input, double* output){
    double det = det_3x3(input);
    output[0] = (input[4]*input[8]-input[7]*input[5]) / det;
    output[3] = (input[6]*input[5]-input[3]*input[8]) / det;
    output[6] = (input[3]*input[7]-input[6]*input[4]) / det;
    
    output[1] = (input[7]*input[2]-input[1]*input[8]) / det;
    output[4] = (input[0]*input[8]-input[6]*input[2]) / det;
    output[7] = (input[1]*input[6]-input[0]*input[7]) / det;

    output[2] = (input[1]*input[5]-input[4]*input[2]) / det;
    output[5] = (input[3]*input[2]-input[0]*input[5]) / det;
    output[8] = (input[0]*input[4]-input[1]*input[3]) / det;
}

__device__ double tetrahedron_volume(const double3 *points)
{
    // Check that we get a tetrahedr
    // Get the coordinates of the four vertices
    const double *x0 = (double *)&(points[0]);
    const double *x1 = (double *)&(points[1]);
    const double *x2 = (double *)&(points[2]);
    const double *x3 = (double *)&(points[3]);

    // Formula for volume from http://mathworld.wolfram.com
    // I see this formula in /dolfin/mesh/TetrahedronCell.cpp which is a part of fenics.
    const double v = (x0[0] * (x1[1] * x2[2] + x3[1] * x1[2] + x2[1] * x3[2] - x2[1] * x1[2] - x1[1] * x3[2] - x3[1] * x2[2])
                    - x1[0] * (x0[1] * x2[2] + x3[1] * x0[2] + x2[1] * x3[2] - x2[1] * x0[2] - x0[1] * x3[2] - x3[1] * x2[2]) 
                    + x2[0] * (x0[1] * x1[2] + x3[1] * x0[2] + x1[1] * x3[2] - x1[1] * x0[2] - x0[1] * x3[2] - x3[1] * x1[2]) 
                    - x3[0] * (x0[1] * x1[2] + x1[1] * x2[2] + x2[1] * x0[2] - x1[1] * x0[2] - x2[1] * x1[2] - x0[1] * x2[2]));

    return std::abs(v) / 6.0;
}

__device__ void get_transformation_operator(const double* p, double* H, double* b, double *inv_H, double* inv_Hb) {

    H[0] = p[3]  - p[0];
    H[1] = p[6]  - p[0];
    H[2] = p[9]  - p[0];

    H[3] = p[4]  - p[1];
    H[4] = p[7]  - p[1];
    H[5] = p[10] - p[1];

    H[6] = p[5]  - p[2];
    H[7] = p[8]  - p[2];
    H[8] = p[11] - p[2];

    inverse_3x3(H, inv_H);

    b[0] = p[0];
    b[1] = p[1];
    b[2] = p[2];

    inv_Hb[0] = - inv_H[0]*p[0] - inv_H[1]*p[1] - inv_H[2]*p[2];
    inv_Hb[1] = - inv_H[3]*p[0] - inv_H[4]*p[1] - inv_H[5]*p[2];
    inv_Hb[2] = - inv_H[6]*p[0] - inv_H[7]*p[1] - inv_H[8]*p[2];
}

__device__ void transform_a_point(double* point_out, const double* point_in, const double* A, const double* b) {

    point_out[0] = A[0*3+0]*point_in[0] + A[0*3+1]*point_in[1] + A[0*3+2]*point_in[2];
    point_out[1] = A[1*3+0]*point_in[0] + A[1*3+1]*point_in[1] + A[1*3+2]*point_in[2];
    point_out[2] = A[2*3+0]*point_in[0] + A[2*3+1]*point_in[1] + A[2*3+2]*point_in[2];

    point_out[0] += b[0];
    point_out[1] += b[1];
    point_out[2] += b[2];
}

// NOTE: How the dofs are arranged?   
// [x,y,z,x,y,z,x,y,z,.....]  FAULSE
// [x,x,...,y,y,...,z,z,...]  TRUE
__device__ void transform_dofs_vector(double* dofs_output, const double* dofs_input){
    for (size_t i = 0; i < 3; i++) // a vector of dimension 3
    {
        dofs_output[i+3*0] =  1.0*dofs_input[i+3*0];
        
        dofs_output[i+3*1] = -3.0*dofs_input[i+3*0] - 1.0*dofs_input[i+3*1] + 4.0*dofs_input[i+3*9];
        dofs_output[i+3*2] = -3.0*dofs_input[i+3*0] - 1.0*dofs_input[i+3*2] + 4.0*dofs_input[i+3*8];
        dofs_output[i+3*3] = -3.0*dofs_input[i+3*0] - 1.0*dofs_input[i+3*3] + 4.0*dofs_input[i+3*7];
        
        dofs_output[i+3*4] =  4.0*dofs_input[i+3*0] + 4.0*dofs_input[i+3*6] - 4.0*dofs_input[i+3*8] - 4.0*dofs_input[i+3*9];
        dofs_output[i+3*5] =  4.0*dofs_input[i+3*0] + 4.0*dofs_input[i+3*5] - 4.0*dofs_input[i+3*7] - 4.0*dofs_input[i+3*9];
        dofs_output[i+3*6] =  4.0*dofs_input[i+3*0] + 4.0*dofs_input[i+3*4] - 4.0*dofs_input[i+3*7] - 4.0*dofs_input[i+3*8];

        dofs_output[i+3*7] =  2.0*dofs_input[i+3*0] + 2.0*dofs_input[i+3*1] - 4.0*dofs_input[i+3*9];
        dofs_output[i+3*8] =  2.0*dofs_input[i+3*0] + 2.0*dofs_input[i+3*2] - 4.0*dofs_input[i+3*8];
        dofs_output[i+3*9] =  2.0*dofs_input[i+3*0] + 2.0*dofs_input[i+3*3] - 4.0*dofs_input[i+3*7];
    }
}

__device__ void transform_dofs_scalar(double* dofs_output, const double* dofs_input){

    dofs_output[0] =  1.0*dofs_input[0];
    
    dofs_output[1] = -3.0*dofs_input[0] - 1.0*dofs_input[1] + 4.0*dofs_input[9];
    dofs_output[2] = -3.0*dofs_input[0] - 1.0*dofs_input[2] + 4.0*dofs_input[8];
    dofs_output[3] = -3.0*dofs_input[0] - 1.0*dofs_input[3] + 4.0*dofs_input[7];
    
    dofs_output[4] =  4.0*dofs_input[0] + 4.0*dofs_input[6] - 4.0*dofs_input[8] - 4.0*dofs_input[9];
    dofs_output[5] =  4.0*dofs_input[0] + 4.0*dofs_input[5] - 4.0*dofs_input[7] - 4.0*dofs_input[9];
    dofs_output[6] =  4.0*dofs_input[0] + 4.0*dofs_input[4] - 4.0*dofs_input[7] - 4.0*dofs_input[8];

    dofs_output[7] =  2.0*dofs_input[0] + 2.0*dofs_input[1] - 4.0*dofs_input[9];
    dofs_output[8] =  2.0*dofs_input[0] + 2.0*dofs_input[2] - 4.0*dofs_input[8];
    dofs_output[9] =  2.0*dofs_input[0] + 2.0*dofs_input[3] - 4.0*dofs_input[7];
}

__device__ void evaluate_vector(const double *dofs, const double *points, double *results, size_t num)
{
    for (size_t i = 0; i < num; i++)
    {
        double x = points[3*i];
        double y = points[3*i+1];
        double z = points[3*i+2];
        for (size_t j = 0; j < 3; j++)
        {
            results[i*3+j] = dofs[j+3*0] + x*dofs[j+3*1]   + y*dofs[j+3*2]   + z*dofs[j+3*3] 
                                            + x*y*dofs[j+3*4] + x*z*dofs[j+3*5] + y*z*dofs[j+3*6]
                                            + x*x*dofs[j+3*7] + y*y*dofs[j+3*8] + z*z*dofs[j+3*9];
        }
    }
}


__global__ void evaluate_vector_function_for_quadrature_points_kernel(
    const double3* function, 
    const size_t* dofmap, 
    const double* quadrature_points,
    double3* results,
    size_t num_gauss,
    size_t num_cells)
{
    // There is no need to know the current position of points when evaluating a function
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index < num_cells)
    // for (size_t index = 0; index < num_cells; index++)
    {
        double *result = (double*)&(results[num_gauss * index]);

        // Get local dofs from function and dofmap
        double3 dof[10];
        for (size_t i = 0; i < 10; i++)
        {
            dof[i] = function[dofmap[index*10+i]];
        }

        // Evaluate at quadrature points
        double dof_params[30];
        transform_dofs_vector(dof_params, (double*)dof);
        evaluate_vector(dof_params, quadrature_points, result, num_gauss);
    }
}


void evaluate_vector_function_for_quadrature_points(
    const double3* function, 
    const size_t* dofmap, 
    const double* quadrature_points,
    double3* results,
    size_t num_gauss,
    size_t num_cells)
{
    std::cout << "\n\n call kernel get_transformation_operator.\n\n" << std::endl;
    uint numThreads, numBlocks;
    computeGridSize(num_cells, 256, numBlocks, numThreads);
    evaluate_vector_function_for_quadrature_points_kernel<<<numBlocks, numThreads>>>(function, dofmap, quadrature_points, results, num_gauss, num_cells);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");
}




}