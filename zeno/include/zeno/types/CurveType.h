#pragma once

#include <string>
#include <magic_enum.hpp>

namespace zeno {

enum struct CurveType {
   
    QUADRATIC_BSPLINE,
    RIBBON_BSPLINE,
    CUBIC_BSPLINE,

    LINEAR,
    BEZIER,
    CATROM
};

static unsigned int CurveDegree(zeno::CurveType type) {

    switch( type ) {
    case CurveType::LINEAR:
        return 1;

    case CurveType::QUADRATIC_BSPLINE:
    case CurveType::RIBBON_BSPLINE:
        return 2;

    case CurveType::CUBIC_BSPLINE:
    case CurveType::BEZIER:
    case CurveType::CATROM:
        return 3;
    }
    return 0;
}

static std::string CurveTypeDefaultString() {
    auto name = magic_enum::enum_name(CurveType::CUBIC_BSPLINE);
    return std::string(name);
}

static std::string CurveTypeListString() {
    auto list = magic_enum::enum_names<CurveType>();

    std::string result;
    for (auto& ele : list) {
        result += " ";
        result += ele;
    }
    return result;
}

} //zeno