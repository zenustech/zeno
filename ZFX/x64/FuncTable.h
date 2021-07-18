#pragma once

#define VCL_NAMESPACE zfx::x64::vcl
#include "vectorclass/vectorclass.h"
#include "vectorclass/vectormath_trig.h"
#include "vectorclass/vectormath_exp.h"
#include <vector>
#include <string>
#include <cmath>

namespace zfx::x64 {
#define DEF_FN1(name) static void func_##name(vcl::Vec4f &x) { x = vcl::name(x); }
#define DEF_FN2(name) static void func_##name(vcl::Vec4f &x, vcl::Vec4f &y) { x = vcl::name(x, y); }
    DEF_FN1(sin)
    DEF_FN1(cos)
    DEF_FN1(tan)
    DEF_FN1(asin)
    DEF_FN1(acos)
    DEF_FN1(atan)
    DEF_FN1(exp)
    DEF_FN1(log)
    DEF_FN1(floor)
    DEF_FN1(ceil)
    DEF_FN1(abs)
    DEF_FN2(min)
    DEF_FN2(max)
    DEF_FN2(atan2)
    DEF_FN2(pow)
#undef DEF_FN1
#undef DEF_FN2

    static void *global_func_table[] = {
#define DEF_FN1(name) (void *)func_##name,
#define DEF_FN2(name) (void *)func_##name,
        DEF_FN1(sin)
        DEF_FN1(cos)
        DEF_FN1(tan)
        DEF_FN1(asin)
        DEF_FN1(acos)
        DEF_FN1(atan)
        DEF_FN1(exp)
        DEF_FN1(log)
        DEF_FN1(floor)
        DEF_FN1(ceil)
        DEF_FN1(abs)
        DEF_FN2(min)
        DEF_FN2(max)
        DEF_FN2(atan2)
        DEF_FN2(pow)
#undef DEF_FN1
#undef DEF_FN2
    };

    static std::vector<std::string> funcnames = {
#define DEF_FN1(name) #name,
#define DEF_FN2(name) #name,
        DEF_FN1(sin)
        DEF_FN1(cos)
        DEF_FN1(tan)
        DEF_FN1(asin)
        DEF_FN1(acos)
        DEF_FN1(atan)
        DEF_FN1(exp)
        DEF_FN1(log)
        DEF_FN1(floor)
        DEF_FN1(ceil)
        DEF_FN1(abs)
        DEF_FN2(min)
        DEF_FN2(max)
        DEF_FN2(atan2)
        DEF_FN2(pow)
#undef DEF_FN1
#undef DEF_FN2
    };
}
