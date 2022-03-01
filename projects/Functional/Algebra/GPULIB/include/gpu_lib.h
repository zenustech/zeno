#include <helper_functions.h>
#include <helper_cuda.h>
#include <double_math.h>

namespace gpu {
// why we use extern "C"
// https://blog.csdn.net/junparadox/article/details/52704108


    /**
     * @brief distribute force from solid to fluid
     * @param solid_forces          variable defined on quadrture points
     * @param quadrature_rules      quadrature points and quadrature weights
     * @param fluid_forces          number of quadrature points
     * @param num                   variable defined on regular mesh
     * @param h                     the spacing of regular mesh
     * @param dim   
     */
    void distribute_force(        
        const double3* solid_forces, 
        const double4* quadrature_rules, 
              double3* fluid_forces, 
        size_t num, double h, int3 dim, bool useCUDA);

    // I do not smile like before.
    // I wish I could be someone you need.
    
    
    // ImmersedBoundaryMethod.cu
    /**
     * @brief 
     * @param solid_velocities 
     * @param quadrature_rules 
     * @param fluid_velocities 
     * @param num 
     * @param h 
     * @param dim 
     */
    void interpolate_velocity(
              double3* solid_velocities, 
        const double4* quadrature_rules,  // weihgts can not be used here.
        const double3* fluid_velocities,
        size_t num, double h, int3 dim, bool useCUDA);

    void gpu_copy(char* dst, char * src, size_t size);

    void gpu_to_cpu(char* dst, char * src, size_t size);

    void cpu_to_gpu(char* dst, char * src, size_t size);

    void gpu_malloc(void ** buffer, size_t size);

    void freeGPUBuffer(void* buffer);

    // GpuVector.cu
    template<class T> T gpu_abs_max(T* a, size_t num);

    template<class T> T gpu_inner(T* a, T* b, size_t num);

    template<class T> void gpu_axpy(T* z, const T* x, const T* y, T a, size_t num);

    template<class T> void gpu_fill(T* x, T a, size_t num);
    
    template<class T> T gpu_sum(T* a, size_t num);
    template<class T> T gpu_min(T* a, size_t num);
    template<class T> T gpu_max(T* a, size_t num);

    void evaluate_vector_function_for_quadrature_points(
        const double3* function, 
        const size_t* dofmap, 
        const double* quadrature_points,
        double3* results,
        size_t num_gauss,
        size_t num_cells);
}