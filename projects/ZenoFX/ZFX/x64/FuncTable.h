#pragma once

#define VCL_NAMESPACE zfx::x64::vcl
#include "vectorclass/vectorclass.h"
#include "vectorclass/vectormath_trig.h"
#include "vectorclass/vectormath_exp.h"
#include <vector>
#include <string>
#include <cmath>

namespace zfx::x64 {

struct FuncTable {
#define DEF_FN1(name) static void func_##name(float *a) { vcl::Vec4f x; x.load(a); x = vcl::name(x); x.store(a); }
#define DEF_FN2(name) static void func_##name(float *a, float *b) { vcl::Vec4f x, y; x.load(a); y.load(b); x = vcl::name(x, y); x.store(a); }
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
DEF_FN2(atan2)
DEF_FN2(pow)
#undef DEF_FN1
#undef DEF_FN2

    static inline std::vector<std::string> funcnames = {
#define DEF_FN1(name) #name,
#define DEF_FN2(name) DEF_FN1(name)
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
DEF_FN2(atan2)
DEF_FN2(pow)
#undef DEF_FN1
#undef DEF_FN2
    };

    std::vector<void *> funcptrs;

    FuncTable() {
        // we have to assign funcptrs at runtime to prevent dll relocation
        for (int i = 0; i < funcnames.size(); i++) {
#define DEF_FN1(name) funcptrs.push_back((void *)func_##name);
#define DEF_FN2(name) DEF_FN1(name)
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
DEF_FN2(atan2)
DEF_FN2(pow)
#undef DEF_FN1
#undef DEF_FN2
        }
    }
};

}
