#pragma once

#include <cstdint>
#include <tuple>
#include <zeno/utils/vec.h>

namespace zeno {

struct PerlinNoise1 {

static inline constexpr int permutation[] = {151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,151,
160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,
37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,
11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,
139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,
46,245,40,244,102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208,89,18,
169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,
43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,
97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,
239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,
254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
};

static float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

static int inc(int num) {
    return num + 1;
}

static float grad(int hash, float x, float y, float z) {
    switch(hash & 0xF)
    {
        case 0x0: return  x + y;
        case 0x1: return -x + y;
        case 0x2: return  x - y;
        case 0x3: return -x - y;
        case 0x4: return  x + z;
        case 0x5: return -x + z;
        case 0x6: return  x - z;
        case 0x7: return -x - z;
        case 0x8: return  y + z;
        case 0x9: return -y + z;
        case 0xA: return  y - z;
        case 0xB: return -y - z;
        case 0xC: return  y + x;
        case 0xD: return -y + z;
        case 0xE: return  y - x;
        case 0xF: return -y - z;
        default: return 0;
    }
}

static float perlin(float x, float y, float z) {
    x = fract(x / 256.f) * 256.f;
    y = fract(y / 256.f) * 256.f;
    z = fract(z / 256.f) * 256.f;
    int xi = (int)x & 255;
    int yi = (int)y & 255;
    int zi = (int)z & 255;
    float xf = x-(int)x;
    float yf = y-(int)y;
    float zf = z-(int)z;
    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);
    int aaa = permutation[permutation[permutation[    xi ]+    yi ]+    zi ];
    int aba = permutation[permutation[permutation[    xi ]+inc(yi)]+    zi ];
    int aab = permutation[permutation[permutation[    xi ]+    yi ]+inc(zi)];
    int abb = permutation[permutation[permutation[    xi ]+inc(yi)]+inc(zi)];
    int baa = permutation[permutation[permutation[inc(xi)]+    yi ]+    zi ];
    int bba = permutation[permutation[permutation[inc(xi)]+inc(yi)]+    zi ];
    int bab = permutation[permutation[permutation[inc(xi)]+    yi ]+inc(zi)];
    int bbb = permutation[permutation[permutation[inc(xi)]+inc(yi)]+inc(zi)];
    float x1 = mix(    grad (aaa, xf  , yf  , zf),
            grad (baa, xf-1, yf  , zf),
            u);
    float x2 = mix(    grad (aba, xf  , yf-1, zf),
            grad (bba, xf-1, yf-1, zf),
            u);
    float y1 = mix(x1, x2, v);
    x1 = mix(    grad (aab, xf  , yf  , zf-1),
            grad (bab, xf-1, yf  , zf-1),
            u);
    x2 = mix(    grad (abb, xf  , yf-1, zf-1),
            grad (bbb, xf-1, yf-1, zf-1),
            u);
    float y2 = mix (x1, x2, v);
    return mix (y1, y2, w);
}

};

struct PerlinNoise {
    static vec3f perlin_hash22(vec3f p)
    {
        p = vec3f( dot(p,vec3f(127.1f,311.7f,284.4f)),
                  dot(p,vec3f(269.5f,183.3f,162.2f)),
                  dot(p,vec3f(228.3f,164.9f,126.0f)));

        return -1.0f + 2.0f * fract(sin(p)*43758.5453123f);
    }

    static float perlin_lev1(vec3f p)
    {
        vec3f pi = floor(p);
        vec3f pf = p - pi;
        vec3f w = pf * pf * (3.0f - 2.0f * pf);
        return 0.08f + 0.8f * (mix(
                    mix(
                        mix(
                            dot(perlin_hash22(pi + vec3f(0, 0, 0)), pf - vec3f(0, 0, 0)),
                            dot(perlin_hash22(pi + vec3f(1, 0, 0)), pf - vec3f(1, 0, 0)),
                            w[0]),
                        mix(
                            dot(perlin_hash22(pi + vec3f(0, 1, 0)), pf - vec3f(0, 1, 0)),
                            dot(perlin_hash22(pi + vec3f(1, 1, 0)), pf - vec3f(1, 1, 0)),
                            w[0]),
                        w[1]),
                    mix(
                        mix(
                            dot(perlin_hash22(pi + vec3f(0, 0, 1)), pf - vec3f(0, 0, 1)),
                            dot(perlin_hash22(pi + vec3f(1, 0, 1)), pf - vec3f(1, 0, 1)),
                            w[0]),
                        mix(
                            dot(perlin_hash22(pi + vec3f(0, 1, 1)), pf - vec3f(0, 1, 1)),
                            dot(perlin_hash22(pi + vec3f(1, 1, 1)), pf - vec3f(1, 1, 1)),
                            w[0]),
                        w[1]),
                    w[2]));
    }

    static float perlin(vec3f a,float power,float depth)
    {
        float total = 0;
        int n = (int)floor(depth);
        for(int i=0; i<n; i++)
        {
            float frequency = 1<<i;
            float amplitude = pow(power,i);
            amplitude *= 1.f - max(0.f, depth - n);
            total += perlin_lev1(a * frequency) * amplitude;
        }

        return total;
    }
};

}
