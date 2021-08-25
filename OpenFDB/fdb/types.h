#pragma once

#include "vec.h"
#include <cstdint>

namespace fdb {

using Qchar = std::int8_t;
using Qshort = std::int16_t;
using Qint = std::int32_t;
using Qlong = std::int64_t;
using Quchar = std::uint8_t;
using Qushort = std::uint16_t;
using Quint = std::uint32_t;
using Qulong = std::uint64_t;
using Qhalf = std::int16_t;
using Qfloat = float;
using Qdouble = double;

using Qchar1 = vec<1, Qchar>;
using Qshort1 = vec<1, Qshort>;
using Qint1 = vec<1, Qint>;
using Qlong1 = vec<1, Qlong>;
using Quchar1 = vec<1, Quchar>;
using Qushort1 = vec<1, Qushort>;
using Quint1 = vec<1, Quint>;
using Qulong1 = vec<1, Qulong>;
using Qhalf1 = vec<1, Qhalf>;
using Qfloat1 = vec<1, Qfloat>;
using Qdouble1 = vec<1, Qdouble>;

using Qchar2 = vec<2, Qchar>;
using Qshort2 = vec<2, Qshort>;
using Qint2 = vec<2, Qint>;
using Qlong2 = vec<2, Qlong>;
using Quchar2 = vec<2, Quchar>;
using Qushort2 = vec<2, Qushort>;
using Quint2 = vec<2, Quint>;
using Qulong2 = vec<2, Qulong>;
using Qhalf2 = vec<2, Qhalf>;
using Qfloat2 = vec<2, Qfloat>;
using Qdouble2 = vec<2, Qdouble>;

using Qchar3 = vec<3, Qchar>;
using Qshort3 = vec<3, Qshort>;
using Qint3 = vec<3, Qint>;
using Qlong3 = vec<3, Qlong>;
using Quchar3 = vec<3, Quchar>;
using Qushort3 = vec<3, Qushort>;
using Quint3 = vec<3, Quint>;
using Qulong3 = vec<3, Qulong>;
using Qhalf3 = vec<3, Qhalf>;
using Qfloat3 = vec<3, Qfloat>;
using Qdouble3 = vec<3, Qdouble>;

using Qchar4 = vec<4, Qchar>;
using Qshort4 = vec<4, Qshort>;
using Qint4 = vec<4, Qint>;
using Qlong4 = vec<4, Qlong>;
using Quchar4 = vec<4, Quchar>;
using Qushort4 = vec<4, Qushort>;
using Quint4 = vec<4, Quint>;
using Qulong4 = vec<4, Qulong>;
using Qhalf4 = vec<4, Qhalf>;
using Qfloat4 = vec<4, Qfloat>;
using Qdouble4 = vec<4, Qdouble>;

}
