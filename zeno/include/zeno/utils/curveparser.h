#pragma once

#include <rapidjson/document.h>
#include <zeno/types/CurveObject.h>

using namespace rapidjson;

namespace zeno {
    CurveData parseCurve(Value const& x, bool& bSucceed);
}