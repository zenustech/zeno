//
// Created by admin on 2022/5/10.
//

#pragma once
#include <iostream>

namespace zfx::x64 {
    struct BuiltFunc {
        #define DEF_FUN1(name) static void func_##name(float *a) {}
        #define DEF_FUN2(name) static void func_##name(float *a , float *b) {}
    };
}