#pragma once
#include "zensim/math/Vec.h"

namespace zeno {

struct ZSPerlinNoise {
    using vec3f = zs::vec<float, 3>;
    static __device__ __host__ vec3f perlin_hash22(vec3f p);

    static __device__ __host__ float perlin_lev1(vec3f p);

    static __device__ __host__ float perlin(vec3f a, float power, float depth);
};

struct ZSPerlinNoise1 {
    static __device__ __host__ float fade(float t);

    static __device__ __host__ int inc(int num);

    static __device__ __host__ float grad(int hash, float x, float y, float z);

    static __device__ __host__ float perlin(float x, float y, float z);
    
    static __device__ __host__ float simplex(float x, float y, float z);
};

} // namespace zeno