#pragma once

#ifndef __CORE_DATA_H__
#define __CORE_DATA_H__

#include <zeno/container/zstring.h>
#include <zeno/container/zvector.h>
#include <zeno/container/zsharedptr.h>
#include <reflect/container/any>
#include <zeno/utils/Error.h>
#include <cassert>

namespace zeno
{
    struct Vec2i {
        int x = 0, y = 0;
        Vec2i() = default;
        Vec2i(int _x, int _y) : x(_x), y(_y) {}
    };

    struct Vec2f {
        float x = 0.f, y = 0.f;
        Vec2f() = default;
        Vec2f(float _x, float _y) : x(_x), y(_y) {}
    };

    struct Vec3i {
        int x = 0, y = 0, z = 0;

        Vec3i() = default;
        Vec3i(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}

        int& operator[](size_t index) {
            assert(index < 3 && index >= 0 && "Vec3i index out of range");
            switch (index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default:
                throw makeError<UnimplError>("index out of range");
            }
        }

        const int& operator[](size_t index) const {
            assert(index < 3 && index >= 0 && "Vec3i index out of range");
            switch (index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default:
                throw makeError<UnimplError>("index out of range");
            }
        }
    };

    struct Vec3f {
        float x = 0.f, y = 0.f, z = 0.f;

        Vec3f() = default;
        Vec3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

        float& operator[](size_t index) {
            assert(index < 3 && index >= 0 && "Vec3i index out of range");
            switch (index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default:
                throw makeError<UnimplError>("index out of range");
            }
        }

        const float& operator[](size_t index) const {
            assert(index < 3 && index >= 0 && "Vec3i index out of range");
            switch (index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default:
                throw makeError<UnimplError>("index out of range");
            }
        }
    };

    struct Vec4i {
        int x, y, z, w;
    };

    struct Vec4f {
        float x, y, z, w;
    };

    struct PrimParamABI {

    };

}

#endif