#pragma once
#ifndef _CUDA_TOOLS_H
#define _CUDA_TOOLS_H

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_SAFE_CALL(err)     cuda_safe_call_(err, __FILE__, __LINE__)
const static int default_threads = 256;
//#define CUDA_KERNEL_CHECK(err)  cuda_kernel_check_(err, __FILE__, __LINE__)

inline unsigned long long LogIte(unsigned long long value) {
    if (value == 0) {
        return 0;
    }
    return 1 + LogIte(value >> 1);
}
inline unsigned long long Log2(unsigned long long value) {
    value -= 1;
    if (value == 0) {
        return 1;
    }
    return LogIte(value);
}


inline void cuda_safe_call_(cudaError err, const char* file_name, const int num_line)
{
    if (cudaSuccess != err)
    {
        std::cerr << file_name << "[" << num_line << "]: "
            << "CUDA Running API error[" << (int)err << "]: "
            << cudaGetErrorString(err) << std::endl;

        exit(0);
    }
}
#endif
