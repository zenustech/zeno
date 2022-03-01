/**
 * @file GpuVector.cu
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2022-01-09
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */


#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

#include <limits>

/// TODO : should there be operations for size_t ?
namespace gpu {

template<typename T>
struct absolute_value : public thrust::unary_function<T,T>
{
    __forceinline__ __host__ __device__ T operator()(const T &x) const
    {
        return x < T(0) ? -x : x;
    }
};

template<class T>
T gpu_abs_max(T* a, size_t num)
{
    return thrust::transform_reduce(thrust::device_ptr<T>(a),
                                    thrust::device_ptr<T>(a+num),
                                    absolute_value<T>(),
                                    0,
                                    thrust::maximum<T>());
}

template int gpu_abs_max<int>(int* a, size_t num);
template float gpu_abs_max<float>(float* a, size_t num);
template double gpu_abs_max<double>(double* a, size_t num);

template<class T>
T gpu_sum(T* a, size_t num)
{
    return thrust::reduce(  thrust::device_ptr<T>(a), 
                            thrust::device_ptr<T>(a+num), 
                            T(0), 
                            thrust::plus<T>());
}

template int gpu_sum<int>(int* a, size_t num);
template ulong gpu_sum<ulong>(ulong* a, size_t num);
template float gpu_sum<float>(float* a, size_t num);
template double gpu_sum<double>(double* a, size_t num);

template<class T>
T gpu_min(T* a, size_t num)
{
    return thrust::reduce(  thrust::device_ptr<T>(a), 
                            thrust::device_ptr<T>(a+num), 
                            std::numeric_limits<T>::max(), 
                            thrust::minimum<T>());
}

template int gpu_min<int>(int* a, size_t num);
template ulong gpu_min<ulong>(ulong* a, size_t num);
template float gpu_min<float>(float* a, size_t num);
template double gpu_min<double>(double* a, size_t num);

template<class T>
T gpu_max(T* a, size_t num)
{
    return thrust::reduce(  thrust::device_ptr<T>(a), 
                            thrust::device_ptr<T>(a+num), 
                            std::numeric_limits<T>::min(), 
                            thrust::maximum<T>());
}

template int gpu_max<int>(int* a, size_t num);
template ulong gpu_max<ulong>(ulong* a, size_t num);
template float gpu_max<float>(float* a, size_t num);
template double gpu_max<double>(double* a, size_t num);

template<typename T>
struct transform_multiple : public thrust::unary_function<thrust::tuple<T, T>,T>
{
    __forceinline__ __host__ __device__ T operator()(const thrust::tuple<T,T> &x) const
    {
        return thrust::get<0>(x)*thrust::get<1>(x);
    }
};

// TODO : use const T* ?
template<class T>
T gpu_inner(T* a, T* b, size_t num)
{
    // use thrust::transform_reduce and thrust::make_zip_iterator to do this job.
    // link for official example :
    // https://github.com/NVIDIA/thrust/blob/383fb9a245fcd61e4b6d50a2bab872d7f1a6cc83/examples/padded_grid_reduction.cu#L21-L39

    return thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<T>(a), thrust::device_ptr<T>(b))),
        thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<T>(a), thrust::device_ptr<T>(b))) + num,
        transform_multiple<T>(),
        T(0),
        thrust::plus<T>());
}

template int gpu_inner<int>(int* a, int* b, size_t num);
template size_t gpu_inner<size_t>(size_t* a, size_t* b, size_t num);
template float gpu_inner<float>(float* a, float* b, size_t num);
template double gpu_inner<double>(double* a, double* b, size_t num);

template<typename T>
struct transform_axpy : public thrust::binary_function<T, T, T>
{
    const T a;

    transform_axpy(T _a) : a(_a) {}
    
    __forceinline__ __host__ __device__ 
    T operator()(const T& x, const T& y) const
    {
        return a*x + y;
    }
};


// z = a*x + y
template<class T>
void gpu_axpy(T* z, const T* x, const T* y, T a, size_t num)
{
    // usage of thrust::transform:
    // Y <- A * X + Y
    // thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));

    // link of example for thrust::transform 
    // https://github.com/NVIDIA/thrust/blob/1669350bcbc026e2df10ab75bbc4f088761024d1/examples/dot_products_with_zip.cu#L99-L106

    thrust::transform( thrust::device_ptr<const T>(x), 
                       thrust::device_ptr<const T>(x+num),
                       thrust::device_ptr<const T>(y),
                       thrust::device_ptr<T>(z),
                       transform_axpy<T>(a) );    
}

template void gpu_axpy<float>(float* z, const float* x, const float* y, float a, size_t num);
template void gpu_axpy<double>(double* z, const double* x, const double* y, double a, size_t num);
template void gpu_axpy<int>(int* z, const int* x, const int* y, int a, size_t num);
template void gpu_axpy<size_t>(size_t* z, const size_t* x, const size_t* y, size_t a, size_t num);


// z = a*x + y
template<class T>
void gpu_fill(T* x, T a, size_t num)
{
    thrust::fill(thrust::device_ptr<T>(x), thrust::device_ptr<T>(x+num), a);
}



template void gpu_fill(double* x, double a, size_t num);
template void gpu_fill(float* x, float a, size_t num);
template void gpu_fill(int* x, int a, size_t num);
template void gpu_fill(size_t* x, size_t a, size_t num);




}