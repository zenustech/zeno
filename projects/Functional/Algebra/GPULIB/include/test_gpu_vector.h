
#include <helper_functions.h>
#include <helper_cuda.h>

namespace gpu{
template<typename T> void gpu_feprojection_form(const T* x, T* Ax, size_t num);


template<typename T>
void gpu_nonlinear_residual(const T* x, T* r, size_t num);






}