#pragma once
#include "zxxglslvec.h"

static __inline__ __device__
vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}
static __inline__ __device__
vec3 ACESFilm(vec3 x)
{
    x = x * vec3(0.539,0.55,0.55);
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return (x*(a*x+b))/(x*(c*x+d)+e);
//    vec3 a = x * (x + 0.0245786f) - 0.000090537f;
//    vec3 b = x * (0.983729f * x + 0.4329510f) + 0.238081f;
//    return a / b;
}

static __inline__ __device__
vec3 ACESFitted(vec3 color, float gamma=1.0f)
{
    // Apply RRT and ODT
    //color = clamp(RRTAndODTFit(color), vec3(0), vec3(1));
    //color = applyLook(color);
    color = mix(vec3(dot(vec3(0.272229, 0.674082, 0.0536895), color)), color, 0.92);
    color = ACESFilm(color);


    vec3 v1 = vec3(1.60475, -0.53108, -0.07367);
    vec3 v2 = vec3(-0.10208, 1.10813, -0.00605);
    vec3 v3 = vec3(-0.00327, -0.07276, 1.07602);
    color = vec3(dot(color, v1), dot(color, v2), dot(color, v3));

    // Clamp to [0, 1]
    color = clamp(color, 0.0f, 1.0f);

    //color = pow(color, vec3(1.0f / gamma));

    return color;
}

static __inline__ __device__
vec3 PhysicalCamera(vec3 in,
                   float aperture = 2,
                   float shutterSpeed = 1.0/25,
                   float iso = 150,
                   float middleGrey = 0.18f,
                   bool enableExposure = true,
                   bool enableACES = true)
{
  vec3 mapped;
  float exposure = middleGrey / ( (1000.0f / 65.0f) * aperture * aperture / (iso * shutterSpeed) );
  mapped = in * exposure;
  return  enableExposure? (enableACES? ACESFilm(mapped):mapped ) : (enableACES? ACESFilm(in) : in);
}
