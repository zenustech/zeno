#pragma once
#include <math.h>
#include <algorithm>
#include <array>
#include <zeno/zeno.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct CubicKernel
{
    using vec3f = zeno::vec3f;

    static float m_radius;
    static float m_k;
    static float m_l;
    static float m_W_zero;

    static void set(float val)
    {
        m_radius = val;
        static const float pi = (M_PI);

        const float h3 = m_radius*m_radius*m_radius;
        m_k = (8.0) / (pi*h3);
        m_l = (48.0) / (pi*h3);
        m_W_zero = W(0.0);
    }

    static float W(const float rl)
    {
        float res = 0.0;
        // const float rl = r.norm();
        // const float rl = length(r);
        const float q = rl/m_radius;
        {
            if (q <= 0.5)
            {
                const float q2 = q*q;
                const float q3 = q2*q;
                res = m_k * ((6.0)*q3- (6.0)*q2+ (1.0));
            }
            else
            {
                res = m_k * ((2.0)*pow((1.0)-q,3));
            }
        }
        return res;
    }

    static vec3f gradW(const vec3f &r)
    {
        vec3f res;
        // const float rl = r.norm();
        const float rl = length(r);
        const float q = rl / m_radius;

        {
            if (rl > 1.0e-6)
            {
                const vec3f gradq = r * ( 1.0 / (rl*m_radius));
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
