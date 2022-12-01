#include "Noise.cuh"

namespace zeno {

__device__ __host__ ZSPerlinNoise::vec3f ZSPerlinNoise::perlin_hash22(vec3f p) {
    p = vec3f(p.dot(vec3f(127.1f, 311.7f, 284.4f)), p.dot(vec3f(269.5f, 183.3f, 162.2f)),
              p.dot(vec3f(228.3f, 164.9f, 126.0f)));
    vec3f ret{};
    for (int d = 0; d != 3; ++d) {
        auto val = zs::sin(p.val(d)) * 43758.5453123f;
        ret.val(d) = -1.0f + 2.0f * (val - zs::floor(val));
    }

    return ret;
}

__device__ __host__ float ZSPerlinNoise::perlin_lev1(vec3f p) {
    vec3f pi = vec3f{zs::floor(p[0]), zs::floor(p[1]), zs::floor(p[2])};
    vec3f pf = p - pi;
    vec3f w = pf * pf * (3.0f - 2.0f * pf);
    return 0.08f + 0.8f * (zs::linear_interop(
                              w[2],
                              zs::linear_interop(
                                  w[1],
                                  zs::linear_interop(w[0], dot(perlin_hash22(pi + vec3f(0, 0, 0)), pf - vec3f(0, 0, 0)),
                                                     dot(perlin_hash22(pi + vec3f(1, 0, 0)), pf - vec3f(1, 0, 0))),
                                  zs::linear_interop(w[0], dot(perlin_hash22(pi + vec3f(0, 1, 0)), pf - vec3f(0, 1, 0)),
                                                     dot(perlin_hash22(pi + vec3f(1, 1, 0)), pf - vec3f(1, 1, 0)))),
                              zs::linear_interop(
                                  w[1],
                                  zs::linear_interop(w[0], dot(perlin_hash22(pi + vec3f(0, 0, 1)), pf - vec3f(0, 0, 1)),
                                                     dot(perlin_hash22(pi + vec3f(1, 0, 1)), pf - vec3f(1, 0, 1))),
                                  zs::linear_interop(w[0], dot(perlin_hash22(pi + vec3f(0, 1, 1)), pf - vec3f(0, 1, 1)),
                                                     dot(perlin_hash22(pi + vec3f(1, 1, 1)), pf - vec3f(1, 1, 1))))));
}

__device__ __host__ float ZSPerlinNoise::perlin(vec3f a, float power, float depth) {
    float total = 0;
    int n = (int)zs::ceil(depth);
    for (int i = 0; i < n; i++) {
        float frequency = 1 << i;
        float amplitude = zs::pow(power, (float)i);
        amplitude *= 1.f - zs::max(0.f, i - (depth - 1));
        total += perlin_lev1(a * frequency) * amplitude;
    }

    return total;
}

} // namespace zeno