#pragma once

#define VCL_NAMESPACE zfx::x64::vcl
#include "vectorclass/vectorclass.h"
#include "vectorclass/vectormath_trig.h"
#include <vector>
#include <string>

namespace zfx::x64 {
    struct my_vec4 {
        float m[4];
    };

    static void func_sin() {
        return;
    }

    static void *global_func_table[] =
        { (void *)func_sin
        };

    static std::vector<std::string> funcnames =
        { "sin"
        };
}
