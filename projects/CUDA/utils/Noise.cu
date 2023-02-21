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

__device__ __host__ float ZSPerlinNoise1::fade(float t) {
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

__device__ __host__ int ZSPerlinNoise1::inc(int num) {
    return num + 1;
}

__device__ __host__ float ZSPerlinNoise1::grad(int hash, float x, float y, float z) {
    switch (hash & 0xF) {
    case 0x0: return x + y;
    case 0x1: return -x + y;
    case 0x2: return x - y;
    case 0x3: return -x - y;
    case 0x4: return x + z;
    case 0x5: return -x + z;
    case 0x6: return x - z;
    case 0x7: return -x - z;
    case 0x8: return y + z;
    case 0x9: return -y + z;
    case 0xA: return y - z;
    case 0xB: return -y - z;
    case 0xC: return y + x;
    case 0xD: return -y + z;
    case 0xE: return y - x;
    case 0xF: return -y - z;
    default: return 0;
    }
}

// Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
__device__ __host__ unsigned int ZSPerlinNoise1::hash(unsigned int seed) {
    seed = (seed + 0x7ed55d16) + (seed << 12);
    seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
    seed = (seed + 0x165667b1) + (seed << 5);
    seed = (seed + 0xd3a2646c) ^ (seed << 9);
    seed = (seed + 0xfd7046c5) + (seed << 3);
    seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

    return seed;
}

// Random value for perlin/simplex noise [0, 255]
__device__ __host__ unsigned char ZSPerlinNoise1::permutation(int p) {
    return (unsigned char)hash(p & 255);
}

__device__ __host__ float ZSPerlinNoise1::perlin(float x, float y, float z) {
#if 0
    constexpr unsigned char hash[] = {
        151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,   225, 140, 36,  103, 30,  69,  142,
        8,   99,  37,  240, 21,  10,  23,  190, 6,   148, 247, 120, 234, 75,  0,   26,  197, 62,  94,  252, 219, 203,
        117, 35,  11,  32,  57,  177, 33,  88,  237, 149, 56,  87,  174, 20,  125, 136, 171, 168, 68,  175, 74,  165,
        71,  134, 139, 48,  27,  166, 77,  146, 158, 231, 83,  111, 229, 122, 60,  211, 133, 230, 220, 105, 92,  41,
        55,  46,  245, 40,  244, 102, 143, 54,  65,  25,  63,  161, 1,   216, 80,  73,  209, 76,  132, 187, 208, 89,
        18,  169, 200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186, 3,   64,  52,  217, 226, 250,
        124, 123, 5,   202, 38,  147, 118, 126, 255, 82,  85,  212, 207, 206, 59,  227, 47,  16,  58,  17,  182, 189,
        28,  42,  223, 183, 170, 213, 119, 248, 152, 2,   44,  154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,
        129, 22,  39,  253, 19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104, 218, 246, 97,  228, 251, 34,
        242, 193, 238, 210, 144, 12,  191, 179, 162, 241, 81,  51,  145, 235, 249, 14,  239, 107, 49,  192, 214, 31,
        181, 199, 106, 157, 184, 84,  204, 176, 115, 121, 50,  45,  127, 4,   150, 254, 138, 236, 205, 93,  222, 114,
        67,  29,  24,  72,  243, 141, 128, 195, 78,  66,  215, 61,  156, 180,
    };
    auto permutation = [&](unsigned int val) { return hash[val & 255]; };
#endif
    auto fract = [](float val) { return val - zs::floor(val); };

    x = fract(x / 256.f) * 256.f;
    y = fract(y / 256.f) * 256.f;
    z = fract(z / 256.f) * 256.f;
    int xi = (int)x & 255;
    int yi = (int)y & 255;
    int zi = (int)z & 255;
    float xf = x - (int)x;
    float yf = y - (int)y;
    float zf = z - (int)z;
    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);
    int aaa = permutation(permutation(permutation(xi) + yi) + zi);
    int aba = permutation(permutation(permutation(xi) + inc(yi)) + zi);
    int aab = permutation(permutation(permutation(xi) + yi) + inc(zi));
    int abb = permutation(permutation(permutation(xi) + inc(yi)) + inc(zi));
    int baa = permutation(permutation(permutation(inc(xi)) + yi) + zi);
    int bba = permutation(permutation(permutation(inc(xi)) + inc(yi)) + zi);
    int bab = permutation(permutation(permutation(inc(xi)) + yi) + inc(zi));
    int bbb = permutation(permutation(permutation(inc(xi)) + inc(yi)) + inc(zi));
    float x1 = zs::linear_interop(u, grad(aaa, xf, yf, zf), grad(baa, xf - 1, yf, zf));
    float x2 = zs::linear_interop(u, grad(aba, xf, yf - 1, zf), grad(bba, xf - 1, yf - 1, zf));
    float y1 = zs::linear_interop(v, x1, x2);
    x1 = zs::linear_interop(u, grad(aab, xf, yf, zf - 1), grad(bab, xf - 1, yf, zf - 1));
    x2 = zs::linear_interop(u, grad(abb, xf, yf - 1, zf - 1), grad(bbb, xf - 1, yf - 1, zf - 1));
    float y2 = zs::linear_interop(v, x1, x2);
    return zs::linear_interop(w, y1, y2);
}

__device__ __host__ float ZSPerlinNoise1::simplex(float x, float y, float z) {
#if 0
    constexpr unsigned char hash[] = {
        151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,   225, 140, 36,  103, 30,  69,  142,
        8,   99,  37,  240, 21,  10,  23,  190, 6,   148, 247, 120, 234, 75,  0,   26,  197, 62,  94,  252, 219, 203,
        117, 35,  11,  32,  57,  177, 33,  88,  237, 149, 56,  87,  174, 20,  125, 136, 171, 168, 68,  175, 74,  165,
        71,  134, 139, 48,  27,  166, 77,  146, 158, 231, 83,  111, 229, 122, 60,  211, 133, 230, 220, 105, 92,  41,
        55,  46,  245, 40,  244, 102, 143, 54,  65,  25,  63,  161, 1,   216, 80,  73,  209, 76,  132, 187, 208, 89,
        18,  169, 200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186, 3,   64,  52,  217, 226, 250,
        124, 123, 5,   202, 38,  147, 118, 126, 255, 82,  85,  212, 207, 206, 59,  227, 47,  16,  58,  17,  182, 189,
        28,  42,  223, 183, 170, 213, 119, 248, 152, 2,   44,  154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,
        129, 22,  39,  253, 19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104, 218, 246, 97,  228, 251, 34,
        242, 193, 238, 210, 144, 12,  191, 179, 162, 241, 81,  51,  145, 235, 249, 14,  239, 107, 49,  192, 214, 31,
        181, 199, 106, 157, 184, 84,  204, 176, 115, 121, 50,  45,  127, 4,   150, 254, 138, 236, 205, 93,  222, 114,
        67,  29,  24,  72,  243, 141, 128, 195, 78,  66,  215, 61,  156, 180,
    };
    auto permutation = [&](unsigned int val) { return hash[val & 255]; };
#endif
    float n0, n1, n2, n3; // Noise contributions from the four corners

    // Skewing/Unskewing factors for 3D
    constexpr float F3 = 1.0f / 3.0f;
    constexpr float G3 = 1.0f / 6.0f;

    // Skew the input space to determine which simplex cell we're in
    float s = (x + y + z) * F3; // Very nice and simple skew factor for 3D
    int i = zs::floor(x + s);
    int j = zs::floor(y + s);
    int k = zs::floor(z + s);
    float t = (i + j + k) * G3;
    float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = x - X0; // The x,y,z distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0; // X Y Z order
        } else if (x0 >= z0) {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 0;
            k2 = 1; // X Z Y order
        } else {
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 1;
            j2 = 0;
            k2 = 1; // Z X Y order
        }
    } else { // x0<y0
        if (y0 < z0) {
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 0;
            j2 = 1;
            k2 = 1; // Z Y X order
        } else if (x0 < z0) {
            i1 = 0;
            j1 = 1;
            k1 = 0;
            i2 = 0;
            j2 = 1;
            k2 = 1; // Y Z X order
        } else {
            i1 = 0;
            j1 = 1;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0; // Y X Z order
        }
    }

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.
    float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    // Work out the hashed gradient indices of the four simplex corners
    int gi0 = permutation(i + permutation(j + permutation(k)));
    int gi1 = permutation(i + i1 + permutation(j + j1 + permutation(k + k1)));
    int gi2 = permutation(i + i2 + permutation(j + j2 + permutation(k + k2)));
    int gi3 = permutation(i + 1 + permutation(j + 1 + permutation(k + 1)));

    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0) {
        n0 = 0.0;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * grad(gi0, x0, y0, z0);
    }
    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0) {
        n1 = 0.0;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * grad(gi1, x1, y1, z1);
    }
    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0) {
        n2 = 0.0;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * grad(gi2, x2, y2, z2);
    }
    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0) {
        n3 = 0.0;
    } else {
        t3 *= t3;
        n3 = t3 * t3 * grad(gi3, x3, y3, z3);
    }
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3);
}

} // namespace zeno