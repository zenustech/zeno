#pragma once
#ifndef HOSTED
#define HOSTED

#ifndef __device__
#define __device__ /* device */
#endif

#ifndef __inline__
#define __inline__ /* inline */
#endif

#ifndef __forceinline__
#define __forceinline__ __inline__
#endif

#include <cmath>
#include <cstdint>
#include <stdint.h>

#ifndef __CUDACC_RTC__

static inline bool isnan(float f) {
    return f == INFINITY;
}

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

#endif // __CUDACC_RTC__
#endif // HOSTED