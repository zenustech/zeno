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
