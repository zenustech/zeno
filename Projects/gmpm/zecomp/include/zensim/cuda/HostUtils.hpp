#pragma once
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <string>

#include "zensim/Reflection.h"

namespace zs {

#define checkThrustErrors(func)                                                            \
  try {                                                                                    \
    func;                                                                                  \
  } catch (thrust::system_error & e) {                                                     \
    std::cout << std::string(__FILE__) << ":" << __LINE__ << " " << e.what() << std::endl; \
  }

  inline static const char *_cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorName(error); }
  template <typename T> inline static std::string _cudaGetErrorEnum(T error) {
    return demangle<T>();
  }

#ifdef __DRIVER_TYPES_H__
#  ifndef DEVICE_RESET
#    define DEVICE_RESET cudaDeviceReset();
#  endif
#else
#  ifndef DEVICE_RESET
#    define DEVICE_RESET
#  endif
#endif

  template <typename T>
  void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
              static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
      DEVICE_RESET
      // Make sure we call CUDA Device Reset before exiting
      exit(EXIT_FAILURE);
    }
  }

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

  inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
      fprintf(stderr,
              "%s(%i) : getLastCudaError() CUDA error :"
              " %s : (%d) %s.\n",
              file, line, errorMessage, static_cast<int>(err), cudaGetErrorString(err));
      DEVICE_RESET
      exit(EXIT_FAILURE);
    }
  }

  template <class T> __inline__ __host__ T *getRawPtr(thrust::device_vector<T> &V) {
    return thrust::raw_pointer_cast(V.data());
  }
  template <class T>
  __inline__ __host__ thrust::device_ptr<T> getDevicePtr(thrust::device_vector<T> &V) {
    return thrust::device_ptr<T>(thrust::raw_pointer_cast(V.data()));
  }
  template <class T> __inline__ __host__ thrust::device_ptr<T> getDevicePtr(T *V) {
    return thrust::device_ptr<T>(V);
  }

  inline void reportMemory(std::string msg) {
    std::size_t free_byte;
    std::size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
      exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage (%s): used = %f, free = %f MB, total = %f MB\n", msg.data(),
           used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
  }

}  // namespace zs
