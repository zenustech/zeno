#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Interpolation.h>

namespace {
using namespace zeno;

static const int permutation[] = {151,160,137,91,90,15,
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


struct VDBAddPerlinNoise : INode {
  virtual void apply() override {
    auto inoutSDF = get_input<VDBFloatGrid>("inoutSDF");
    auto strength = get_input<NumericObject>("strength")->get<float>();
    auto scale = get_input<NumericObject>("scale")->get<float>();
    auto scaling = has_input("scaling") ?
        get_input<NumericObject>("scaling")->get<vec3f>()
        : vec3f(1);
    auto translation = has_input("translation") ?
        get_input<NumericObject>("translation")->get<vec3f>()
        : vec3f(0);
    auto inv_scale = 1.f / (scale * scaling);

    auto grid = inoutSDF->m_grid;
    float dx = grid->voxelSize()[0];
    strength *= dx;

    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto coord = iter.getCoord();
            auto pos = (vec3i(coord[0], coord[1], coord[2]) + translation) * inv_scale;
            auto noise = strength * perlin(pos[0], pos[1], pos[2]);
            iter.modifyValue([&] (auto &v) {
                v += noise;
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);

    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

ZENO_DEFNODE(VDBAddPerlinNoise)(
     { /* inputs: */ {
     "inoutSDF",
     {"float", "strength", "1.0"},
     {"float", "scale", "8.0"},
     {"vec3f", "scaling", "1,1,1"},
     {"vec3f", "translation", "0,0,0"},
     }, /* outputs: */ {
       "inoutSDF",
     }, /* params: */ {
     }, /* category: */ {
     "deprecated",
     }});




template <class T0, class T1, class T2>
auto smoothstep(T0 edge0, T1 edge1, T2 x) {
  // Scale, bias and saturate x to 0..1 range
  x = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  // Evaluate polynomial
  return x * x * (3 - 2 * x);
}


struct Turbulent {

using vec3 = vec3f;

// Noise settings:
//float Power = 5.059;
//float MaxLength = 0.9904;
//float Dumping = 10.0;

vec3 hash3(vec3 p) {
	p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
			dot(p, vec3(269.5, 183.3, 246.1)),
			dot(p, vec3(113.5, 271.9, 124.6)));

	return -1.0f + 2.0f * fract(sin(p) * 43758.5453123f);
}

float noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);

	vec3 u = f * f * (3.0 - 2.0 * f);

	float n0 = dot(hash3(i + vec3(0.0, 0.0, 0.0)), f - vec3(0.0, 0.0, 0.0));
	float n1 = dot(hash3(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0));
	float n2 = dot(hash3(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0));
	float n3 = dot(hash3(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0));
	float n4 = dot(hash3(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0));
	float n5 = dot(hash3(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0));
	float n6 = dot(hash3(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0));
	float n7 = dot(hash3(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0));

	float ix0 = mix(n0, n1, u[0]);
	float ix1 = mix(n2, n3, u[0]);
	float ix2 = mix(n4, n5, u[0]);
	float ix3 = mix(n6, n7, u[0]);

	float ret = mix(mix(ix0, ix1, u[1]), mix(ix2, ix3, u[1]), u[2]) * 0.5 + 0.5;
	return ret * 2.0 - 1.0;
}


/*float distToObject(vec2 p) {
	p *= 0.2;

	vec2 newSeed = vec2(iTime * TIME_FACTOR + 1.0);
    newSeed.y *= 0.2;
    newSeed = floor(newSeed);
    newSeed *= 4.0;

	return rune(p, newSeed - 0.41);
}

float normalizeScalar(float value, float max) {
	return clamp(value, 0.0, max) / max;
}*/

float color(vec3 coord) {
	float n = abs(noise(coord));
	n += 0.5 * abs(noise(coord * 2.0));
	n += 0.25 * abs(noise(coord * 4.0));
	n += 0.125 * abs(noise(coord * 8.0));

	//n *= (100.001 - Power);
	//float dist = distToObject(p);
	//float k = normalizeScalar(dist, MaxLength);
	//n *= dist / pow(1.001 - k, Dumping);

	return n;
}

float operator()(float x, float y, float z) {  // https://www.shadertoy.com/view/MtSSRz
    return color({x, y, z});
}

};

struct VDBAddTurbulentNoise : INode {
  virtual void apply() override {
    auto inoutSDF = get_input<VDBFloatGrid>("inoutSDF");
    auto strength = get_input<NumericObject>("strength")->get<float>();
    auto scale = get_input<NumericObject>("scale")->get<float>();
    auto scaling = has_input("scaling") ?
        get_input<NumericObject>("scaling")->get<vec3f>()
        : vec3f(1);
    auto translation = has_input("translation") ?
        get_input<NumericObject>("translation")->get<vec3f>()
        : vec3f(0);
    auto inv_scale = 1.f / (scale * scaling);

    auto grid = inoutSDF->m_grid;
    float dx = grid->voxelSize()[0];
    strength *= dx;

    Turbulent turbulent;

    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            auto coord = iter.getCoord();
            auto pos = (vec3i(coord[0], coord[1], coord[2]) + translation) * inv_scale;
            auto noise = strength * turbulent(pos[0], pos[1], pos[2]);
            iter.modifyValue([&] (auto &v) {
                v += noise;
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);

    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

ZENO_DEFNODE(VDBAddTurbulentNoise)(
     { /* inputs: */ {
     "inoutSDF",
     {"float", "strength", "3.0"},
     {"float", "scale", "16.0"},
     {"vec3f", "scaling", "1,1,1"},
     {"vec3f", "translation", "0,0,0"},
     }, /* outputs: */ {
       "inoutSDF",
     }, /* params: */ {
//{"float","Power","5.059"},
//{"float","MaxLength","0.9904"},
//{"float","Dumping","10.0"},
     }, /* category: */ {
     "openvdb",
     }});


}
