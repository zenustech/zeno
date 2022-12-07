#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/DummyObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

#include "zxxglslvec.h"

float4* devptr;
bool initialized = false;
struct ray {
	float3 origin;
	float3 direction;
};
struct sphere {
	vec3 origin;
	float radius;
	int material;
};
struct hit_record {
	float t;
	int material_id;
	vec3 normal;
	vec3 origin;
};

static __inline__ __device__ float hash( float n )
{
    return fract(sin(n)*43758.5453);
}


static __inline__ __device__ float noise( vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0 + 113.0*p.z;

    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

static __inline__ __device__ float fbm( vec3 p , int layer=6 )
{
    float f = 0.0;
    mat3 m = mat3( 0.00,  0.80,  0.60,
                  -0.80,  0.36, -0.48,
                  -0.60, -0.48,  0.64 );
    vec3 pp = p;
    float coef = 0.5;
    for(int i=0;i<layer;i++) {
        f += coef * noise(pp);
        pp = m * pp *2.02;
        coef *= 0.5;
    }
    return f/0.9375;
}

static __inline__ __device__ void intersect_sphere(
	ray r,
	sphere s,
    hit_record& hit
){
	vec3 oc = s.origin - r.origin;
    float a  = dot(r.direction, r.direction);
	float b = 2 * dot(oc, r.direction);
	float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b*b - 4*a*c;
	if (discriminant < 0) return;

    float t = (-b - sqrt(discriminant) ) / (2.0*a);

	hit.t = t;
	hit.material_id = s.material;
	hit.origin = r.origin + t * r.direction;
	hit.normal = (hit.origin - s.origin) / s.radius;
}

static __inline__ __device__ float density(vec3 pos, vec3 windDir, float coverage, float t)
{
	// signal
	vec3 p = pos * .0212242 + windDir * t; // test time
	float dens = fbm(p); //, FBM_FREQ);;

	float cov = 1. - coverage;
	dens *= smoothstep (cov, cov + .05, dens);

	return clamp(dens, 0., 1.);	
}

static __inline__ __device__ float light(
	vec3 origin,
    vec3 sunLightDir,
    vec3 windDir,
    float coverage,
    float absorption,
    float t
){
	const int steps = 8;
	float march_step = 1.;

	vec3 pos = origin;
	vec3 dir_step = -sunLightDir * march_step; // reverse shadow?
	float T = 1.; // transmitance

	for (int i = 0; i < steps; i++) {
		float dens = density(pos, windDir, coverage, t);

		float T_i = exp(-absorption * dens * march_step);
		T *= T_i;
		//if (T < .01) break;

		pos += dir_step;
	}

	return T;
}

#define sun_color vec3(1., .7, .55)
static __inline__ __device__ vec3 render_sky_color(vec3 rd, vec3 sunLightDir)
{
	double sun_amount = max(dot(rd, normalize(sunLightDir)), 0.0);
	vec3 sky = mix(vec3(.0, .1, .4), vec3(.3, .6, .8), 1.0 - rd.y);
	sky = sky + sun_color * min(pow(sun_amount, 1500.0) * 5.0, 1.0);
	sky = sky + sun_color * min(pow(sun_amount, 10.0) * .6, 1.0);
	return sky;
}

#define SIMULATE_LIGHT
#define FAKE_LIGHT
#define max_dist 1e8
static __inline__ __device__ vec4 render_clouds(
    ray r, 
    vec3 sunLightDir,
    vec3 windDir, 
    int steps, 
    float coverage, 
    float thickness, 
    float absorption, 
    float t
){
    vec3 C = vec3(0, 0, 0);
    float alpha = 0.;

    float march_step = thickness / float(steps);
	vec3 dir_step = r.direction / r.direction.y * march_step;

    sphere atmosphere = {
        vec3(0,-3350, 0), 
        3500., 
        0
    };
    hit_record hit = {
        float(max_dist + 1e1),  // 'infinite' distance
        -1,                     // material id
        vec3(0., 0., 0.),       // normal
        vec3(0., 0., 0.)        // origin
    };

    intersect_sphere(r, atmosphere, hit);
	vec3 pos = hit.origin;

    float T = 1.; // transmitance
	for (int i = 0; i < steps; i++) {
		float h = float(i) / float(steps);
		float dens = density(pos, windDir, coverage, t);

		float T_i = 
            pow(0.99f, smoothstep(0.0f,0.1f,dens)) * 
            exp(-absorption * dens * march_step);
		T *= T_i;
		if (T < .01) break;
        float C_i = 
            T * 
#ifdef SIMULATE_LIGHT
			light(
                pos, 
                sunLightDir,
                windDir,
                coverage,
                absorption,
                t
            ) * 
#endif
			dens * 
            march_step;
        C += C_i;
        alpha += (1. - T_i) * (1. - alpha);
		pos += dir_step;
		if (length(pos) > 1e3) break;
	}

    return vec4(C.x, C.y, C.z, alpha);
}

static __inline__ __device__ float4 proceduralSky(
    vec3 dir, 
    vec3 sunLightDir, 
    vec3 windDir,
    int step,
    float coverage, 
    float thickness,
    float absorption,
    float time
){
    vec3 col = vec3(0,0,0);

    float3 r_dir = normalize(dir);
    ray r = {vec3(0,0,0), r_dir};
    
    vec3 sky = render_sky_color(r.direction, sunLightDir);
    if(r_dir.y<-0.001) return {sky.x, sky.y, sky.z, 0.0f}; // need to upgrade

    vec4 cld = render_clouds(r, sunLightDir, windDir, step, coverage, thickness, absorption, time);
    col = mix(sky, vec3(cld)/(0.000001+cld.w), cld.w);
    return {col.x, col.y, col.z, 0.0f};;
    // return {sky.x, sky.y, sky.z, 0.0f};
}

__global__ 
void pskycu(
    float4* data, 
    int nx,
    int ny
    // to be added
    ,
    float3 sunLightDir,
    float3 windDir,
    int step, // be careful
    float coverage, 
    float thickness,
    float absorption,
    float time
)
{
    // int index = threadIdx.x;
    // int stride = blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int idx=index;idx<(nx*ny);idx+=stride) 
    {
        // need to have texture coordinates (figured out from idx)
        float u = static_cast<float>(idx%nx)/ny;
        float v = static_cast<float>(idx/nx)/ny;
        float2 uv = make_float2(u, v);
        float4 col;
        float phi = uv.x*2*3.1415926;
        float theta = uv.y*3.1415926;
        float3 dir = make_float3(-cos(phi)*sin(theta), -cos(theta), sin(phi)*sin(theta));

        // instead of rewriting everything
        // modify to use proceduralsky from DisneyBSDF.h
        // Compile and debug!!!!!!!
        // col = make_float4((dir.x+1.0)*0.5, (dir.y+1.0)*0.5, (dir.z+1.0)*0.5, 0.0);
        
        // col = proceduralSky(dir, make_float3(1,1,1));

        // tofix: upside down
        // float3 sunLightDir = make_float3(
        //     sin(-45.0f / 180.f * M_PI),
        //     cos(-45.0f / 180.f * M_PI) * sin(60.0f / 180.f * M_PI),
        //     cos(-45.0f / 180.f * M_PI) * cos(60.0f / 180.f * M_PI)
        // );

        col = 
        // proceduralSky(
        //     dir,
        //     sunLightDir,
        //     make_float3(0., 0., 1.),
        //     40, // be careful
        //     .45,
        //     15.,
        //     1.030725 * 0.3,
        //     0
        // );
        proceduralSky(
            dir,
            sunLightDir,
            windDir,
            step,
            coverage,
            thickness,
            absorption,
            time
        );

        data[idx] = col;
    }
} 

int cuda_iDivUp(int a, int b) {
    return (a + (b - 1)) / b;
}

extern "C" {
    void initTextureDevData(int nx, int ny)
    {
        if(!initialized) {
            cudaMalloc((void**)&devptr, nx*ny*sizeof(float4));
            initialized = true;
        }
    }

    void computeTexture(
        float4* texptr,
        int nx,
        int ny
        ,
        float3 sunLightDir,
        float3 windDir,
        int step, // be careful
        float coverage, 
        float thickness,
        float absorption,
        float time
    )
    {
        // 1,1 for test
        dim3 block(64, 1, 1);
        dim3 grid(cuda_iDivUp(nx*ny, block.x), 1, 1);
        pskycu<<<grid, block>>>(devptr, nx, ny
            ,
            sunLightDir,
            windDir,
            step,
            coverage,
            thickness,
            absorption,
            time
        );
        cudaDeviceSynchronize();
        cudaMemcpy(texptr, devptr, nx*ny*sizeof(float4), cudaMemcpyDeviceToHost);
    }

    // destructor? to free device memory  
    void freeTexture()
    {
        if(initialized) {
            cudaFree(devptr);
            initialized = false;
        }
    }
}

namespace zeno {
/// utilities
constexpr std::size_t count_warps(std::size_t n) noexcept {
    return (n + 31) / 32;
}
constexpr int warp_index(int n) noexcept {
    return n / 32;
}
constexpr auto warp_mask(int i, int n) noexcept {
    int k = n % 32;
    const int tail = n - k;
    if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
    return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
}

template <typename T, typename Op> __forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst, Op op) {
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val = op(val, tmp);
    }
    if (locid == 0)
        dst = val;
}

template <typename TransOp, typename ReduceOp>
float prim_reduce(typename ZenoParticles::particles_t &verts, float e, TransOp top, ReduceOp rop,
                  std::string attrToReduce) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using T = typename ZenoParticles::particles_t::value_type;
    auto nchn = verts.getPropertySize(attrToReduce);
    auto offset = verts.getPropertyOffset(attrToReduce);
    const auto nwarps = count_warps(verts.size());

    auto cudaPol = cuda_exec().device(0);

    Vector<float> res{verts.get_allocator(), nwarps};
    // cudaPol(res, [e] ZS_LAMBDA(auto &v) { v = e; });
    cudaPol(range(verts.size()), [res = proxy<space>(res), verts = proxy<space>({}, verts), offset, nwarps, nchn, top,
                                  rop] ZS_LAMBDA(int i) mutable {
        auto [mask, numValid] = warp_mask(i, nwarps);
        auto locid = threadIdx.x & 31;
        float v = top(verts(offset, i));
        while (--nchn) {
            v = rop(top(verts(offset++, i)), v);
        }
        reduce_to(i, nwarps, v, res[i / 32], rop);
    });

    Vector<float> ret{res.get_allocator(), 1};
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), e, rop);
    return ret.getVal();
}

struct ZSPrimitiveReduction : zeno::INode {
    struct pass_on {
        template <typename T> constexpr T operator()(T v) const noexcept {
            return v;
        }
    };
    struct getabs {
        template <typename T> constexpr T operator()(T v) const noexcept {
            return zs::abs(v);
        }
    };
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto prim = get_input<ZenoParticles>("ZSParticles");
        auto &verts = prim->getParticles();
        auto attrToReduce = get_input2<std::string>("attr");
        if (attrToReduce == "pos")
            attrToReduce = "x";
        if (attrToReduce == "vel")
            attrToReduce = "v";

        if (!verts.hasProperty(attrToReduce))
            throw std::runtime_error(fmt::format("verts do not have property [{}]\n", attrToReduce));

        auto opStr = get_input2<std::string>("op");
        zeno::NumericValue result;
        if (opStr == "avg") {
            result = prim_reduce(verts, 0, pass_on{}, std::plus<float>{}, attrToReduce) / verts.size();
        } else if (opStr == "max") {
            result = prim_reduce(verts, limits<float>::lowest(), pass_on{}, getmax<float>{}, attrToReduce);
        } else if (opStr == "min") {
            result = prim_reduce(verts, limits<float>::max(), pass_on{}, getmin<float>{}, attrToReduce);
        } else if (opStr == "absmax") {
            result = prim_reduce(verts, 0, getabs{}, getmax<float>{}, attrToReduce);
        }

        auto out = std::make_shared<zeno::NumericObject>();
        out->set(result);
        set_output("result", std::move(out));
    }
};
ZENDEFNODE(ZSPrimitiveReduction, {/* inputs: */ {
                                      "ZSParticles",
                                      {"string", "attr", "pos"},
                                      {"enum avg max min absmax", "op", "avg"},
                                  },
                                  /* outputs: */
                                  {
                                      "result",
                                  },
                                  /* params: */
                                  {},
                                  /* category: */
                                  {
                                      "primitive",
                                  }});

struct ZSGetUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input<ZenoParticles>("object");
        auto key = get_param<std::string>("key");
        auto hasValue = object->userData().has(key);
        auto data = hasValue ? object->userData().get(key) : std::make_shared<DummyObject>();
        set_output2("hasValue", hasValue);
        set_output("data", std::move(data));
    }
};

ZENDEFNODE(ZSGetUserData, {
                              {"object"},
                              {"data", {"bool", "hasValue"}},
                              {{"string", "key", ""}},
                              {"lifecycle"},
                          });

} // namespace zeno