#pragma once

#ifndef __device__
#define __device__ /* device */
#endif

#ifndef __inline__
#define __inline__ /* inline */
#endif

#ifndef __forceinline__
#define __forceinline__ __inline__
#endif

#ifndef __CUDACC_RTC__

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>

#include <cstdint>
#include <stdint.h>

// static inline bool isnan(float f) {
//     return f == INFINITY;
// }

template<typename InType, typename OutType>
static inline OutType bitConvert(InType inValue) {
    OutType* outP = (OutType*)&inValue;
    OutType outV = *outP;
    return outV;
}

static inline int __float_as_int(float f) {
    return bitConvert<float, int>(f);
}

static inline float __int_as_float(int i) {
    return bitConvert<int, float>(i);;
}

static inline uint __float_as_uint(float f) {
    return bitConvert<float, uint>(f);
}

static inline float __uint_as_float(uint i) {
    return bitConvert<uint, float>(i);;
}

// #ifndef isnan
// static inline bool isnan(float v) {
//     return std::isnan(v);
// }
// #endif

// #ifndef isinf
// static inline bool isinf(float v) {
//     return std::isinf(v);
// }
// #endif

#endif // __CUDACC_RTC__