#pragma once

#define VCL_NAMESPACE zfx::x64::vcl
#include "vectorclass/vectorclass.h"
#include "vectorclass/vectormath_trig.h"
#include <vector>
#include <string>

namespace zfx::x64 {
    static vcl::Vec4f func_sin(vcl::Vec4f x) { return vcl::sin(x); }
    static vcl::Vec4f func_cos(vcl::Vec4f x) { return vcl::cos(x); }
    static vcl::Vec4f func_tan(vcl::Vec4f x) { return vcl::tan(x); }

    static void *global_func_table[] =
        { (void *)func_sin
        , (void *)func_cos
        , (void *)func_tan
        };

    static std::vector<std::string> funcnames =
        { "sin"
        , "cos"
        , "tan"
        };
}
