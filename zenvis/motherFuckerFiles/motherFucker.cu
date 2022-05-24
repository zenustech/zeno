
#include <optix.h>

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};


struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    ParallelogramLight     light; // TODO: make light list
    OptixTraversableHandle handle;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float3  emission_color;
    float3  diffuse_color;
    float4* vertices;
};
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}




//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    float        opacity;
    float        prob;
    float        prob2;
    unsigned int seed;
    unsigned int flags = 0;
    int          countEmitted;
    int          done;
    int          pad;
};


struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

namespace zeno{
__forceinline__ __device__ float lerp(float a, float b, float c)
{
    return (1-c)*a + c*b;
}
__forceinline__ __device__ float3 lerp(float3 a, float3 b, float c)
{
    float3 coef = make_float3(c,c,c);
    return (make_float3(1,1,1)-coef)*a + coef*b;
}
__forceinline__ __device__ float length(float3 vec){
    return sqrtf(dot(vec,vec));
}
__forceinline__ __device__ float3 normalize(float3 vec){
    return vec/(zeno::length(vec)+0.00001);
}
__forceinline__ __device__ float clamp(float c, float a, float b){
    return max(min(c,b),a);
}
__forceinline__ __device__ float3 sin(float3 a){
    return make_float3(sinf(a.x), sinf(a.y), sinf(a.z));
}
__forceinline__ __device__ float3 fract(float3 a){
    float3 temp;
    return make_float3(modff(a.x, &temp.x), modff(a.y, &temp.y), modff(a.z, &temp.z));
}
};
namespace BRDFBasics{
__forceinline__ __device__  float fresnel(float cosT){
    float v = zeno::clamp(1-cosT,0.0f,1.0f);
    float v2 = v *v;
    return v2 * v2 * v;
}
__forceinline__ __device__  float GTR1(float cosT,float a){
    if(a >= 1.0) return 1/M_PIf;
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a-1.0f) / (M_PIf*logf(a*a)*t);
}
__forceinline__ __device__  float GTR2(float cosT,float a){
    float t = (1+(a*a-1)*cosT*cosT);
    return (a*a) / (M_PIf*t*t);
}
__forceinline__ __device__  float GGX(float cosT, float a){
    float a2 = a*a;
    float b = cosT*cosT;
    return 1.0/ (cosT + sqrtf(a2 + b - a2*b));
}
__forceinline__ __device__  float3 sampleOnHemisphere(unsigned int &seed, float roughness)
{
    float x = rnd(seed);
    float y = rnd(seed);

    float a = roughness*roughness;
	
	float phi = 2.0 * M_PIf * x;
	float cosTheta = sqrtf((1.0 - y) / (1.0 + (a*a - 1.0) * y));
	float sinTheta = sqrtf(1.0 - cosTheta*cosTheta);
	

    return make_float3(cos(phi) * sinTheta,  sin(phi) * sinTheta, cosTheta);
}
};
namespace DisneyBRDF
{   
__forceinline__ __device__ float pdf(
        float3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float3 N,
        float3 T,
        float3 B,
        float3 wi,
        float3 wo)
    {
        float3 n = N;
        float spAlpha = max(0.001, roughness);
        float ccAlpha = zeno::lerp(0.1, 0.001, clearcoatGloss);
        float diffRatio = 0.5*(1.0 - metallic);
        float spRatio = 1.0 - diffRatio;

        float3 half = zeno::normalize(wi + wo);

        float cosTheta = abs(dot(n, half));
        float pdfGTR2 = BRDFBasics::GTR2(cosTheta, spAlpha) * cosTheta;
        float pdfGTR1 = BRDFBasics::GTR1(cosTheta, ccAlpha) * cosTheta;

        float ratio = 1.0/(1.0 + clearcoat);
        float pdfSpec = zeno::lerp(pdfGTR1, pdfGTR2, ratio)/(4.0 * abs(dot(wi, half)));
        float pdfDiff = abs(dot(wi, n)) * (1.0/M_PIf);

        return diffRatio * pdfDiff + spRatio * pdfSpec;
    }

__forceinline__ __device__ float3 sample_f(
        unsigned int &seed, 
        float3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float3 N,
        float3 T,
        float3 B,
        float3 wo,
        float &is_refl)
    {
        
        float ratiodiffuse = (1.0 - metallic)/2.0;
        float p = rnd(seed);
        
        Onb tbn = Onb(N);
        
        float3 wi;
        
        if( p < ratiodiffuse){
            //sample diffuse lobe
            
            float3 P = BRDFBasics::sampleOnHemisphere(seed, 1.0);
            wi = P;
            tbn.inverse_transform(wi);
            wi = normalize(wi);
            is_refl = 0;
        }else{
            //sample specular lobe.
            float a = max(0.001, roughness);
            
            float3 P = BRDFBasics::sampleOnHemisphere(seed, a*a);
            float3 half = normalize(P);
            tbn.inverse_transform(half);            
            wi = half* 2.0* dot(normalize(wo), half) - normalize(wo); //reflection vector
            wi = normalize(wi);
            is_refl = 1;
        }
        
        return wi;
    }
__forceinline__ __device__ float3 eval(
        float3 baseColor,
        float metallic,
        float subsurface,
        float specular,
        float roughness,
        float specularTint,
        float anisotropic,
        float sheen,
        float sheenTint,
        float clearcoat,
        float clearcoatGloss,
        float3 N,
        float3 T,
        float3 B,
        float3 wi,
        float3 wo)
    {
        float3 wh = normalize(wi+ wo);
        float ndoth = dot(N, wh);
        float ndotwi = dot(N, wi);
        float ndotwo = dot(N, wo);
        float widoth = dot(wi, wh);

        if(ndotwi <=0 || ndotwo <=0 )
            return make_float3(0,0,0);

        float3 Cdlin = baseColor;
        float Cdlum = 0.3*Cdlin.x + 0.6*Cdlin.y + 0.1*Cdlin.z;

        float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : make_float3(1.0,1.0,1.0);
        float3 Cspec0 = zeno::lerp(specular*0.08*zeno::lerp(make_float3(1,1,1), Ctint, specularTint), Cdlin, metallic);
        float3 Csheen = zeno::lerp(make_float3(1.0,1.0,1.0), Ctint, sheenTint);

        //diffuse
        float Fd90 = 0.5 + 2.0 * ndoth * ndoth * roughness;
        float Fi = BRDFBasics::fresnel(ndotwi);
        float Fo = BRDFBasics::fresnel(ndotwo);
        
        float Fd = (1 +(Fd90-1)*Fi)*(1+(Fd90-1)*Fo);

        float Fss90 = widoth*widoth*roughness;
        float Fss = zeno::lerp(1.0, Fss90, Fi) * zeno::lerp(1.0,Fss90, Fo);
        float ss = 1.25 * (Fss *(1.0 / (ndotwi + ndotwo) - 0.5) + 0.5);

        float a = max(0.001, roughness);
        float Ds = BRDFBasics::GTR2(ndoth, a);
        float Dc = BRDFBasics::GTR1(ndoth, zeno::lerp(0.1, 0.001, clearcoatGloss));

        float roughg = sqrtf(roughness*0.5 + 0.5);
        float Gs = BRDFBasics::GGX(ndotwo, roughg) * BRDFBasics::GGX(ndotwi, roughg);

        float Gc = BRDFBasics::GGX(ndotwo, 0.25) * BRDFBasics::GGX(ndotwi, 0.25);

        float Fh = BRDFBasics::fresnel(widoth);
        float3 Fs = zeno::lerp(Cspec0, make_float3(1.0,1.0,1.0), Fh);
        float Fc = zeno::lerp(0.04, 1.0, Fh);

        float3 Fsheen = Fh * sheen * Csheen;

        return ((1/M_PIf) * zeno::lerp(Fd, ss, subsurface) * Cdlin + Fsheen) * (1.0 - metallic)
        + Gs*Fs*Ds + 0.25*clearcoat*Gc*Fc*Dc;
    }
};

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------
///here inject common code
__forceinline__ __device__ float3 perlin_hash22(float3 p)
{
    p = make_float3( dot(p,make_float3(127.1,311.7,284.4)),
              dot(p,make_float3(269.5,183.3,162.2)),
	      	  dot(p,make_float3(228.3,164.9,126.0)));
    float a;
    return -1.0 + 2.0 * zeno::fract(zeno::sin(p)*43758.5453123);
}

__forceinline__ __device__ float perlin_lev1(float3 p)
{
    float3 pi = make_float3(floor(p.x), floor(p.y), floor(p.z));
    float3 pf = p - pi;
    float3 w = pf * pf * (3.0 - 2.0 * pf);
    return .08 + .8 * (zeno::lerp(
			    zeno::lerp(
				    zeno::lerp(
					    dot(perlin_hash22(pi + make_float3(0, 0, 0)), pf - make_float3(0, 0, 0)),
					    dot(perlin_hash22(pi + make_float3(1, 0, 0)), pf - make_float3(1, 0, 0)),
					    w.x),
				    zeno::lerp(
					    dot(perlin_hash22(pi + make_float3(0, 1, 0)), pf - make_float3(0, 1, 0)),
					    dot(perlin_hash22(pi + make_float3(1, 1, 0)), pf - make_float3(1, 1, 0)),
					    w.x),
				    w.y),
			    zeno::lerp(
				    zeno::lerp(
					    dot(perlin_hash22(pi + make_float3(0, 0, 1)), pf - make_float3(0, 0, 1)),
					    dot(perlin_hash22(pi + make_float3(1, 0, 1)), pf - make_float3(1, 0, 1)),
					    w.x),
				    zeno::lerp(
					    dot(perlin_hash22(pi + make_float3(0, 1, 1)), pf - make_float3(0, 1, 1)),
					    dot(perlin_hash22(pi + make_float3(1, 1, 1)), pf - make_float3(1, 1, 1)),
					    w.x),
				    w.y),
			    w.z));
}

__forceinline__ __device__ float perlin(float p,int n,float3 a)
{
    float total = 0;
    for(int i=0; i<n; i++)
    {
        float frequency = powf(2.0,i*1.0f);
        float amplitude = powf(p,i*1.0f);
        total = total + perlin_lev1(a * frequency) * amplitude;
    }

    return total;
}

///


static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}


static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<unsigned int>( occluded ) );
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1 );
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_OCCLUSION,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            RAY_TYPE_OCCLUSION       // missSBTIndex
            );

}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );

    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

        const float2 d = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                ) - 1.0f;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.emitted      = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.prob         = 1.0f;
        prd.prob2        = 1.0f;
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;
        prd.opacity      = 0;
        int depth = 0;
        for( ;; )
        {
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd );

            result += prd.emitted;
            result += prd.radiance * prd.attenuation/prd.prob;

            if( prd.done  || depth >= 5 ) // TODO RR, variable for depth
                break;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;
            if(prd.opacity<0.99)
                ++depth;
        }
    }
    while( --i );

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;
    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3( rt_data->bg_color );
    prd->done      = true;
}


extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
extern "C" __global__ void __anyhit__shadow_cutout()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();
    float pnoise3 = perlin(10.0, 5, P*0.005);
    float mat_opacity = zeno::clamp(pnoise3, 0.0,1.0);

    float opacity = mat_opacity;//sin(P.y)>0?1.0:0.0;
    prd->opacity = opacity;
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99 ) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
    }
    else
    {
        prd->flags |= 1;
        optixTerminateRay();
    }
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();

    if( prd->countEmitted )
        prd->emitted = rt_data->emission_color;
    else
        prd->emitted = make_float3( 0.0f );


    float3 mat_baseColor = make_float3(1.0,0.766,0.336);
    float mat_metallic = 1;
    float mat_roughness = 0.1;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearCoat = 0.0;
    float mat_clearCoatGloss = 0.0;
    float mat_opacity = 0.0;
    //here shall send in "material code"
     mat_baseColor = make_float3(1.0,0.766,0.336);
    mat_metallic = 1;
    mat_roughness = 0.1;
    mat_subsurface = 0.0;
    mat_specular = 0;
    mat_specularTint = 0.0;
    mat_anisotropic = 0.0;
    mat_sheen = 0.0;
    mat_sheenTint = 0.0;
    mat_clearCoat = 0.0;
    mat_clearCoatGloss = 0.0;
    mat_opacity = 0.0;

    float pnoise = perlin(1, 3, P*0.02);
    pnoise = clamp(pnoise, 0.0f, 1.0f);

    float pnoise2 = perlin(1, 4, P*0.02);
    mat_metallic = pnoise;

    mat_roughness = pnoise2;
    mat_roughness = zeno::clamp(mat_roughness, 0.01,0.99)*0.5;

    float pnoise3 = perlin(10.0, 5, P*0.005);
    mat_opacity = zeno::clamp(pnoise3, 0.0,1.0);

    //here shall send in "material code"
    

    

    float opacity = mat_opacity;//sin(P.y)>0?1.0:0.0;
    
    //end of material computation
    mat_metallic = zeno::clamp(mat_metallic,0.01, 0.99);
    mat_roughness = zeno::clamp(mat_roughness, 0.01,0.99);
    //discard fully opacity pixels
    prd->opacity = opacity;
    if(opacity>0.99)
    {
        prd->radiance += make_float3(0.0f);
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }

    //{
    unsigned int seed = prd->seed;
    float is_refl;
    float3 wi = DisneyBRDF::sample_f(
                                seed, 
                                mat_baseColor,
                                mat_metallic,
                                mat_subsurface,
                                mat_specular,
                                mat_roughness,
                                mat_specularTint,
                                mat_anisotropic,
                                mat_sheen,
                                mat_sheenTint,
                                mat_clearCoat,
                                mat_clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                -normalize(ray_dir),
                                is_refl);

    float pdf = DisneyBRDF::pdf(mat_baseColor,
                                mat_metallic,
                                mat_subsurface,
                                mat_specular,
                                mat_roughness,
                                mat_specularTint,
                                mat_anisotropic,
                                mat_sheen,
                                mat_sheenTint,
                                mat_clearCoat,
                                mat_clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    float3 f = DisneyBRDF::eval(mat_baseColor,
                                mat_metallic,
                                mat_subsurface,
                                mat_specular,
                                mat_roughness,
                                mat_specularTint,
                                mat_anisotropic,
                                mat_sheen,
                                mat_sheenTint,
                                mat_clearCoat,
                                mat_clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    prd->prob2 = prd->prob;
    prd->prob *= pdf; 
    prd->origin = P;
    prd->direction = wi;
    prd->countEmitted = false;
    if(is_refl)
        prd->attenuation *= f * clamp(dot(wi, N),0.0f,1.0f);
    else
        prd->attenuation *= f * clamp(dot(wi, N),0.0f,1.0f);
    //}

    // {
    //     const float z1 = rnd(seed);
    //     const float z2 = rnd(seed);

    //     float3 w_in;
    //     cosine_sample_hemisphere( z1, z2, w_in );
    //     Onb onb( N );
    //     onb.inverse_transform( w_in );
    //     prd->direction = w_in;
    //     prd->origin    = P;

    //     prd->attenuation *= rt_data->diffuse_color;
    //     prd->countEmitted = false;
    // }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd->seed = seed;

    ParallelogramLight light = params.light;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const float3 L     = normalize(light_pos - P );
    const float  nDl   = dot( N, L );
    const float  LnDl  = -dot( light.normal, L );

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        prd->flags = 0;
        traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
            );
        unsigned int occluded = prd->flags;
        if( !occluded )
        {
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd->radiance += light.emission * weight;
}
