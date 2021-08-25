#pragma once

#include "vec.h"

namespace fdb {

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using half = decltype([] () { struct { short i; } r; return r; });

using char1 = vec<1, char>;
using short1 = vec<1, short>;
using int1 = vec<1, int>;
using long1 = vec<1, long>;
using uchar1 = vec<1, uchar>;
using ushort1 = vec<1, ushort>;
using uint1 = vec<1, uint>;
using ulong1 = vec<1, ulong>;
using half1 = vec<1, half>;
using float1 = vec<1, float>;
using double1 = vec<1, double>;

using char2 = vec<2, char>;
using short2 = vec<2, short>;
using int2 = vec<2, int>;
using long2 = vec<2, long>;
using uchar2 = vec<2, uchar>;
using ushort2 = vec<2, ushort>;
using uint2 = vec<2, uint>;
using ulong2 = vec<2, ulong>;
using half2 = vec<2, half>;
using float2 = vec<2, float>;
using double2 = vec<2, double>;

using char3 = vec<3, char>;
using short3 = vec<3, short>;
using int3 = vec<3, int>;
using long3 = vec<3, long>;
using uchar3 = vec<3, uchar>;
using ushort3 = vec<3, ushort>;
using uint3 = vec<3, uint>;
using ulong3 = vec<3, ulong>;
using half3 = vec<3, half>;
using float3 = vec<3, float>;
using double3 = vec<3, double>;

using char4 = vec<4, char>;
using short4 = vec<4, short>;
using int4 = vec<4, int>;
using long4 = vec<4, long>;
using uchar4 = vec<4, uchar>;
using ushort4 = vec<4, ushort>;
using uint4 = vec<4, uint>;
using ulong4 = vec<4, ulong>;
using half4 = vec<4, half>;
using float4 = vec<4, float>;
using double4 = vec<4, double>;

}
