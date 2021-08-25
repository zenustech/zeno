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

using Qchar1 = vec<Qchar, 1>;
using Qshort1 = vec<Qshort, 1>;
using Qint1 = vec<Qint, 1>;
using Qlong1 = vec<Qlong, 1>;
using Quchar1 = vec<Quchar, 1>;
using Qushort1 = vec<Qushort, 1>;
using Quint1 = vec<Quint, 1>;
using Qulong1 = vec<Qulong, 1>;
using Qhalf1 = vec<Qhalf, 1>;
using Qfloat1 = vec<Qfloat, 1>;
using Qdouble1 = vec<Qdouble, 1>;

using Qchar2 = vec<Qchar, 2>;
using Qshort2 = vec<Qshort, 2>;
using Qint2 = vec<Qint, 2>;
using Qlong2 = vec<Qlong, 2>;
using Quchar2 = vec<Quchar, 2>;
using Qushort2 = vec<Qushort, 2>;
using Quint2 = vec<Quint, 2>;
using Qulong2 = vec<Qulong, 2>;
using Qhalf2 = vec<Qhalf, 2>;
using Qfloat2 = vec<Qfloat, 2>;
using Qdouble2 = vec<Qdouble, 2>;

using Qchar3 = vec<Qchar, 3>;
using Qshort3 = vec<Qshort, 3>;
using Qint3 = vec<Qint, 3>;
using Qlong3 = vec<Qlong, 3>;
using Quchar3 = vec<Quchar, 3>;
using Qushort3 = vec<Qushort, 3>;
using Quint3 = vec<Quint, 3>;
using Qulong3 = vec<Qulong, 3>;
using Qhalf3 = vec<Qhalf, 3>;
using Qfloat3 = vec<Qfloat, 3>;
using Qdouble3 = vec<Qdouble, 3>;

using Qchar4 = vec<Qchar, 4>;
using Qshort4 = vec<Qshort, 4>;
using Qint4 = vec<Qint, 4>;
using Qlong4 = vec<Qlong, 4>;
using Quchar4 = vec<Quchar, 4>;
using Qushort4 = vec<Qushort, 4>;
using Quint4 = vec<Quint, 4>;
using Qulong4 = vec<Qulong, 4>;
using Qhalf4 = vec<Qhalf, 4>;
using Qfloat4 = vec<Qfloat, 4>;
using Qdouble4 = vec<Qdouble, 4>;

}
