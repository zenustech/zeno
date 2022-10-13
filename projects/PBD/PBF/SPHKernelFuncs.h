#pragma once
#include <zeno/zeno.h>
//SPH kernel function
inline float kernelPoly6(float dist, float h)
{
    float coeff = 315.0 / 64.0 / 3.14159265358979323846;
    float res = 0.0;
    if(dist > 0 && dist < h)
    {
        float x = (h * h - dist * dist) / (h * h * h);
        res = coeff * x * x * x;
    }
    return res;
}

//SPH kernel gradient
inline zeno::vec3f kernelSpikyGradient(const zeno::vec3f& r, float h)
{
    float coeff = -45.0 / 3.14159265358979323846;
    zeno::vec3f res{0.0, 0.0, 0.0};
    float dist = length(r);
    if (dist > 0 && dist < h)
    {
        float x = (h - dist) / (h * h * h);
        float factor = coeff * x * x;
        res = r * factor / dist;
    }
    return res;
}