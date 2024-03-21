#pragma once
#include "Sampling.h"
#include "zxxglslvec.h"
#include "TraceStuff.h"

static __inline__ __device__ float hash( float n )
{
    return fract(sin(n)*43758.5453f);
}


static __inline__ __device__ float noise( vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0f-2.0*f);

    float n = p.x + p.y*57.0f + 113.0f*p.z;

    float res = mix(mix(mix( hash(n+  0.0f), hash(n+  1.0f),f.x),
                        mix( hash(n+ 57.0f), hash(n+ 58.0f),f.x),f.y),
                    mix(mix( hash(n+113.0f), hash(n+114.0f),f.x),
                        mix( hash(n+170.0f), hash(n+171.0f),f.x),f.y),f.z);
    return res;
}

static __inline__ __device__ float fbm( vec3 p , int layer=6)
{
    float f = 0.0;
    mat3 m = mat3( 0.00f,  0.80f,  0.60f,
                  -0.80f,  0.36f, -0.48f,
                  -0.60f, -0.48f,  0.64f );
    vec3 pp = p;
    float coef = 0.5f;
    for(int i=0;i<layer;i++) {
        f += coef * noise(pp);
        pp = m * pp *2.02f;
        coef *= 0.5f;
    }
    return f/0.9375f;
}
static __inline__ __device__
    mat3 rot(float deg){
    return mat3(cos(deg),-sin(deg),0,
                sin(deg), cos(deg),0,
                0,0,1);

}

static __inline__ __device__ vec3 proceduralSky2(vec3 dir, vec3 sunLightDir, float t)
{

    float bright = 1*(1.8f-0.55f);
    float color1 = fbm((dir*3.5f)-0.5f);  //xz
    float color2 = fbm((dir*7.8f)-10.5f); //yz

    float clouds1 = smoothstep(1.0f-0.55f,fmin((1.0f-0.55f)+0.28f*2.0f,1.0f),color1);
    float clouds2 = smoothstep(1.0f-0.55f,fmin((1.0f-0.55f)+0.28f,1.0f),color2);

    float cloudsFormComb = saturate(clouds1+clouds2);
    vec3 sunCol = vec3(258.0, 208.0, 100.0) / 15.0f;

    vec4 skyCol = vec4(0.6,0.8,1.0,1.0);
    float cloudCol = saturate(saturate(1.0-pow(color1,1.0f)*0.2f)*bright);
    vec4 clouds1Color = vec4(cloudCol,cloudCol,cloudCol,1.0f);
    vec4 clouds2Color = mix(clouds1Color,skyCol,0.25f);
    vec4 cloudColComb = mix(clouds1Color,clouds2Color,saturate(clouds2-clouds1));
    vec4 clouds = vec4(0.0);
    clouds = mix(skyCol,cloudColComb,cloudsFormComb);

    vec3 localRay = normalize(dir);
    float sunIntensity = 1.0f - (dot(localRay, sunLightDir) * 0.5f + 0.5f);
    sunIntensity = 0.2f / sunIntensity;
    sunIntensity = fmin(sunIntensity, 40000.0f);
    sunIntensity = fmax(0.0f, sunIntensity - 3.0f);
    //return vec3(0,0,0);
    return vec3(clouds)*0.5f + sunCol * (sunIntensity*0.0000075f);
}

// ####################################
#define sun_color vec3(1.f, .7f, .55f)
static __inline__ __device__ vec3 render_sky_color(vec3 rd, vec3 sunLightDir)
{
	float sun_amount = fmax(dot(rd, normalize(sunLightDir)), 0.0f);
	vec3 sky = mix(vec3(.0f, .1f, .4f), vec3(.3f, .6f, .8f), 1.0f - rd.y);
	sky = sky + sun_color * fmin(powf(sun_amount, 1500.0f) * 5.0f, 1.0f);
	sky = sky + sun_color * fmin(powf(sun_amount, 10.0f) * .6f, 1.0f);
	return sky;
}
struct ray {
	vec3 origin;
	vec3 direction;
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

    float t = (-b - sqrt(discriminant) ) / (2.0f*a);

	hit.t = t;
	hit.material_id = s.material;
	hit.origin = r.origin + t * r.direction;
	hit.normal = (hit.origin - s.origin) / s.radius;
}
static __inline__ __device__ float softlight(float base, float blend, float c)
{
    return (blend < c) ? (2.0 * base * blend + base * base * (1.0f - 2.0f * blend)) : (sqrt(base) * (2.0f * blend - 1.0f) + 2.0f * base * (1.0f - blend));
}
static __inline__ __device__ float density(vec3 pos, vec3 windDir, float coverage, float t, float freq = 1.0f, int layer = 6)
{
	// signal
	vec3 p = 2.0f *  pos * .0212242f * freq; // test time
        vec3 pertb = vec3(noise(p*16), noise(vec3(p.x,p.z,p.y)*16), noise(vec3(p.y, p.x, p.z)*16)) * 0.05f;
	float dens = fbm(p + pertb + windDir * t, layer); //, FBM_FREQ);;

	float cov = 1.f - coverage;
//	dens = smoothstep (cov-0.1, cov + .1, dens);
//        dens = softlight(fbm(p*4 + pertb * 4  + windDir * t), dens, 0.8);
        dens *= smoothstep (cov, cov + .1f, dens);
	return pow(clamp(dens, 0.f, 1.f),0.5f);
}
static __inline__ __device__ float light(
	vec3 origin,
    vec3 sunLightDir,
    vec3 windDir,
    float coverage,
    float absorption,
    float t,
    float freq = 1.0
){
	const int steps = 4;
	float march_step = 0.5;

	vec3 pos = origin;
	vec3 dir_step = -sunLightDir * march_step;
	float T = 1.; // transmitance
        float coef = 1.0;
	for (int i = 0; i < steps; i++) {
		float dens = density(pos, windDir, coverage, t, freq,6);

		float T_i = exp(-absorption * dens * coef * march_step);
		T *= T_i;
		//if (T < .01) break;

		pos = vec3(
            pos.x + coef * dir_step.x,
            pos.y + coef * dir_step.y,
            pos.z + coef * dir_step.z
        );
            coef *= 2.0f;
	}

	return T;
}
#define SIMULATE_LIGHT
#define FAKE_LIGHT
#define max_dist 1e8f
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
    //r.direction.x = r.direction.x * 2.0f;
    vec3 C = vec3(0, 0, 0);
    float alpha = 0.;
    float s = mix(30, 10, sqrtf(r.direction.y));
    float march_step = thickness / floor(s) / 2;
    vec3 dir_step = r.direction / sqrtf(r.direction.y)  * march_step ;

    sphere atmosphere = {
        vec3(0,-350, 0),
        500., 
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
    float talpha = 0;
    float T = 1.; // transmitance
    float coef = 1.0;
    for (int i =0; i < int(s)/2; i++)
    {
        float freq = mix(0.5f, 1.0f, smoothstep(0.0f, 0.5f, r.direction.y));
        float dens = density(pos, windDir, coverage, t, freq);
        dens = mix(0.0f,dens, smoothstep(0.0f, 0.2f, r.direction.y));
        float T_i = exp(-absorption * dens * coef *  2.0f* march_step);
        T *= T_i;
        if (T < .01f)
            break;
        talpha += (1.f - T_i) * (1.f - talpha);
        pos = vec3(
            pos.x + coef * 2.0f* dir_step.x,
            pos.y + coef * 2.0f* dir_step.y,
            pos.z + coef * 2.0f* dir_step.z
        );
        coef *= 1.0f;
        if (length(pos) > 1e3f) break;
    }

        //vec3 pos = r.direction * 500.0f;
    pos = hit.origin;
        alpha = 0;
        T = 1.; // transmitance
        coef = 1.0;
    if (talpha > 1e-3f) {
        for (int i = 0; i < int(s); i++) {
            float h = float(i) / float(steps);
            float freq = mix(0.5f, 1.0f, smoothstep(0.0f, 0.5f, r.direction.y));
            float dens = density(pos, windDir, coverage, t, freq);
            dens = mix(0.0f, dens, smoothstep(0.0f, 0.2f, r.direction.y));
            float T_i =

                exp(-absorption * dens * coef * march_step);
            T *= T_i;
            if (T < .01f)
                break;
            float C_i;

                C_i = T *
#ifdef SIMULATE_LIGHT
                      light(pos, sunLightDir, windDir, coverage, absorption, t, freq) *
#endif
                      // #ifdef FAKE_LIGHT
                      // 			(exp(h) / 1.75) *
                      // #endif
                      dens * march_step;

                C = vec3(C.x + C_i, C.y + C_i, C.z + C_i);
                alpha += (1.f - T_i) * (1.f - alpha);
                pos = vec3(pos.x + coef * dir_step.x,
                           pos.y + coef * dir_step.y,
                           pos.z + coef * dir_step.z);
                coef *= 1.0f;
                if (length(pos) > 1e3f)
                    break;
            }
        }
    return vec4(C.x, C.y, C.z, alpha);
}

static __inline__ __device__ vec3 proceduralSky(
    vec3 dir, 
    vec3 sunLightDir, 
    vec3 windDir,
    int steps,
    float coverage, 
    float thickness,
    float absorption,
    float t
){
    vec3 col = vec3(0,0,0);

    vec3 r_dir = normalize(dir);
    ray r = {vec3(0,0,0), r_dir};
    
    vec3 sky = render_sky_color(r.direction, sunLightDir);
    if(r_dir.y<-0.001f) return sky; // just being lazy

    vec4 cld = render_clouds(r, sunLightDir, windDir, steps, coverage, thickness, absorption, t);
    col = mix(sky, vec3(cld)/(0.000001f+cld.w), cld.w);
    return col;
}

static __inline__ __device__ vec3 hdrSky2(
    vec3 dir
){
  dir = dir
            .rotY(to_radians(params.sky_rot_y))
            .rotX(to_radians(params.sky_rot_x))
            .rotZ(to_radians(params.sky_rot_z))
            .rotY(to_radians(params.sky_rot));
            
  vec3 uv = sphereUV(dir, true);
  vec3 col = vec3(0);
  for(int jj=-2;jj<=2;jj++)
  {
    for(int ii=-2;ii<=2;ii++)
    {
      float dx = (float)ii / (float)(params.skynx);
      float dy = (float)jj / (float)(params.skyny);
      col = col + (vec3)texture2D(params.sky_texture, vec2(uv[0] + dx, uv[1] + dy)) * params.sky_strength;
    }
  }

  return col/9.0f;
}

static __inline__ __device__ vec3 hdrSky(
        vec3 dir, float upperBound,  float isclamp, float &pdf
){
    dir = dir
            .rotY(to_radians(params.sky_rot_y))
            .rotX(to_radians(params.sky_rot_x))
            .rotZ(to_radians(params.sky_rot_z))
            .rotY(to_radians(params.sky_rot));

    vec3 uv = sphereUV(dir, true);

    vec3 col = (vec3)texture2D(params.sky_texture, vec2(uv[0], uv[1]));//* params.sky_strength;
    vec3 col2 = clamp(col, vec3(0.0f), vec3(upperBound));
    int i = uv[0] * params.skynx;
    int j = uv[1] * params.skyny;
    //float p = params.skycdf[params.skynx * params.skyny + j * params.skynx + i];
    pdf = luminance(col) / params.envavg  / (2.0f * M_PIf * M_PIf);
    return mix(col, col2, isclamp) * params.sky_strength;
}

static __inline__ __device__ vec3 colorTemperatureToRGB(float temperatureInKelvins)
{
    vec3 retColor;

    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0f, 40000.0f) / 100.0f;

    if (temperatureInKelvins <= 66.0f)
    {
        retColor.x = 1.0f;
        retColor.y = saturate(0.39008157876901960784f * log(temperatureInKelvins) - 0.63184144378862745098f);
    }
    else
    {
        float t = temperatureInKelvins - 60.0f;
        retColor.x = saturate(1.29293618606274509804f * pow(t, -0.1332047592f));
        retColor.y = saturate(1.12989086089529411765f * pow(t, -0.0755148492f));
    }

    if (temperatureInKelvins >= 66.0f)
        retColor.z = 1.0;
    else if(temperatureInKelvins <= 19.0f)
        retColor.z = 0.0;
    else
        retColor.z = saturate(0.54320678911019607843f * log(temperatureInKelvins - 10.0f) - 1.19625408914f);

    return retColor;
}
static __inline__ __device__ vec3 envSky2(vec3 dir)
{
  return hdrSky2(dir);
}
static __inline__ __device__ vec3 envSky(
    vec3 dir,
    vec3 sunLightDir,
    vec3 windDir,
    int steps,
    float coverage,
    float thickness,
    float absorption,
    float t,
    float &pdf,
    float upperBound = 100.0f,
    float isclamp = 0.0f
){
    vec3 color;
    if (!params.usingHdrSky) {
        color = proceduralSky(
            dir,
            sunLightDir,
            windDir,
            steps,
            coverage,
            thickness,
            absorption,
            t
        );
    }
    else {
        color = hdrSky(
            dir, upperBound, isclamp, pdf
        );
    }
    if (params.colorTemperatureMix > 0) {
        vec3 colorTemp = colorTemperatureToRGB(params.colorTemperature);
        colorTemp = mix(vec3(1, 1, 1), colorTemp, params.colorTemperatureMix);
        color = color * colorTemp;
    }
    return color;
}
