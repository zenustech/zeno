#pragma once

#include <array>
#include <cmath>

#include <zeno/utils/vec.h>

struct CubicKernel
{

    // static float m_k;
    // static float m_l;
    // static float m_W_zero;
    // float pi = (3.14159265358979323846);
    // static void set(float h=0.1)
    // {
    //     const float pi = (3.14159265358979323846);
    //     const float h3 = h*h*h;
    //     m_k = (8.0) / (pi*h3);
    //     m_l = (48.0) / (pi*h3);
    //     m_W_zero = W(0.0);
    // }
    static float length(zeno::vec3f r)
    {
        return std::hypot(r[0],r[1],r[2]);
    }
    
    static float W(const float rl, const float h = 0.1)
    {
        const float pi = (3.14159265358979323846);
        const float h3 = h*h*h;
        float m_k = (8.0) / (pi*h3);

        float res = 0.0;
        const float q = rl/h;
        {
            if (q <= 0.5)
            {
                const float q2 = q*q;
                const float q3 = q2*q;
                res = m_k * ((6.0)*q3- (6.0)*q2+ (1.0));
            }
            else
            {
                res = m_k * ((2.0)*std::pow((1.0)-q,3));
            }
        }
        return res;
    }

    //重载vec3f版本
    static float W(const zeno::vec3f r, const float h = 0.1)
    {
        const float pi = (3.14159265358979323846);
        const float h3 = h*h*h;
        float m_k = (8.0) / (pi*h3);

        float res = 0.0;
        float rl = zeno::length(r);
        const float q = rl/h;
        {
            if (q <= 0.5)
            {
                const float q2 = q*q;
                const float q3 = q2*q;
                res = m_k * ((6.0)*q3- (6.0)*q2+ (1.0));
            }
            else
            {
                res = m_k * ((2.0)*std::pow((1.0)-q,3));
            }
        }
        return res;
    }


    static zeno::vec3f gradW(const zeno::vec3f &r, const float h = 0.1)
    {
        const float pi = (3.14159265358979323846);
        const float h3 = h*h*h;
        const float m_l = (48.0) / (pi*h3);

        zeno::vec3f res;
        // const float rl = r.norm();
        const float rl = zeno::length(r);
        const float q = rl / h;

        {
            if (rl > 1.0e-6)
            {
                const zeno::vec3f gradq = r * ( 1.0 / (rl*h));
                if (q <= 0.5)
                {
                    res = m_l*q*( 3.0*q -  2.0)*gradq;
                }
                else
                {
                    const float factor = (1.0) - q;
                    res = m_l*(-factor*factor)*gradq;
                }
            }
        }
        return res;
    }
};


struct Poly6Kernel
{

    static float W(float dist, float h=0.1)
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
};

struct SpikyKernel
{
    static zeno::vec3f gradW(const zeno::vec3f& r, float h=0.1)
    {
        float coeff = -45.0 / 3.14159265358979323846;
        zeno::vec3f res{0.0, 0.0, 0.0};
        float dist = zeno::length(r);
        if (dist > 0 && dist < h)
        {
            float x = (h - dist) / (h * h * h);
            float factor = coeff * x * x;
            res = r * factor / dist;
        }
        return res;
    }
};