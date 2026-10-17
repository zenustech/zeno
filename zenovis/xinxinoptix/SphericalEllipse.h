#pragma once

#include <sutil/vec_math.h>

#ifndef ZENO_ELLIPSE_USE_LUT
// LUT-assisted inversion is available for comparison. Measurements do not
// establish a consistent GPU speedup, so preserve Newton as the default.
#define ZENO_ELLIPSE_USE_LUT 0
#endif
#include <SphericalEllipseLut.h>

// Independent implementation of the polar map in Guillén et al.,
// "Area-Preserving Parameterizations for Spherical Ellipses", EGSR 2017.
// https://iliyan.com/publications/DiskSampling/
//
// Diagonalizing the ellipse's cone also supports non-circular planar emitters.
// In its principal frame the spherical boundary is
//   (a cos(t), b sin(t), sqrt(1 - a*a*cos(t)^2 - b*b*sin(t)^2)).
// Its polar sector area has the well-conditioned derivative
//   dOmega/dt = a*b / (1 + boundary.z).
// Numerical integration/inversion of this expression avoids subtracting two
// nearly equal elliptic integrals. An optional inverse-CDF LUT seeds the same
// checked Newton inversion; it does not change the sampling PDF or light UVs.
struct SphericalEllipse {
    float3 x, y, z;
    float a, b, cosA2, cosB2;
    float quadrantIntegral;
    float solidAngle;

    SUTIL_HOSTDEVICE static inline void rotate(float q[3][3], float v[3][3], int p, int r) {
        const float off = q[p][r];
        if (fabsf(off) <= 1.0e-7f * (fabsf(q[p][p]) + fabsf(q[r][r]))) return;
        const float tau = (q[r][r] - q[p][p]) / (2.0f * off);
        const float t = copysignf(1.0f / (fabsf(tau) + sqrtf(1.0f + tau * tau)), tau);
        const float c = 1.0f / sqrtf(1.0f + t * t);
        const float s = t * c;
        q[p][p] -= t * off;
        q[r][r] += t * off;
        q[p][r] = q[r][p] = 0.0f;
        for (int k = 0; k < 3; ++k) {
            if (k != p && k != r) {
                const float kp = q[k][p], kr = q[k][r];
                q[k][p] = q[p][k] = c * kp - s * kr;
                q[k][r] = q[r][k] = s * kp + c * kr;
            }
            const float vp = v[k][p], vr = v[k][r];
            v[k][p] = c * vp - s * vr;
            v[k][r] = s * vp + c * vr;
        }
    }

    SUTIL_HOSTDEVICE inline float density(float t) const {
        const float s = sinf(t), c = cosf(t);
        return 1.0f / (1.0f + sqrtf(fmaxf(0.0f, cosA2 * c * c + cosB2 * s * s)));
    }

    SUTIL_HOSTDEVICE inline float integral(float upper) const {
        // Eight-point Gauss-Legendre integration on [0, upper].
        const float nodes[4] = {0.1834346425f, 0.5255324099f, 0.7966664774f, 0.9602898565f};
        const float weights[4] = {0.3626837834f, 0.3137066459f, 0.2223810345f, 0.1012285363f};
        const float half = 0.5f * upper;
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i)
            sum += weights[i] * (density(half * (1.0f - nodes[i])) +
                                 density(half * (1.0f + nodes[i])));
        return half * sum;
    }

    SUTIL_HOSTDEVICE inline bool init(const float3& centerDelta,
                                      const float3& axisX, float radiusX,
                                      const float3& axisY, float radiusY,
                                      float minSolidAngle = 3.0e-4f) {
        solidAngle = 0.0f;
        if (!(radiusX > 0.0f && radiusY > 0.0f)) return false;
        const float3 normal = cross(axisX, axisY);
        const float distance2 = dot(centerDelta, centerDelta);
        const float planeDistance = dot(centerDelta, normal);
        if (!(distance2 > 0.0f) || !isfinite(distance2) || planeDistance == 0.0f) return false;
        const float approximateAngle = 3.14159265358979323846f * radiusX * radiusY *
                                       (fabsf(planeDistance) / distance2) / sqrtf(distance2);
        // Uniform disk-area sampling is better conditioned for distant lights.
        if (approximateAngle < minSolidAngle) return false;

        float cx = dot(centerDelta, axisX) / radiusX;
        float cy = dot(centerDelta, axisY) / radiusY;
        float dx = planeDistance / radiusX, dy = planeDistance / radiusY;
        const float scale = fmaxf(1.0f, fmaxf(fmaxf(fabsf(cx), fabsf(cy)),
                                             fmaxf(fabsf(dx), fabsf(dy))));
        if (!isfinite(scale)) return false;
        const float invScale = 1.0f / scale;
        cx *= invScale; cy *= invScale; dx *= invScale; dy *= invScale;
        const float q00 = dx * dx, q11 = dy * dy;
        float q[3][3] = {{q00, 0.0f, -dx * cx},
                          {0.0f, q11, -dy * cy},
                          {-dx * cx, -dy * cy, cx * cx + cy * cy - invScale * invScale}};
        float vectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int sweep = 0; sweep < 4; ++sweep) {
            rotate(q, vectors, 0, 1);
            rotate(q, vectors, 0, 2);
            rotate(q, vectors, 1, 2);
        }
        int negative = 0;
        if (q[1][1] < q[negative][negative]) negative = 1;
        if (q[2][2] < q[negative][negative]) negative = 2;
        int major = (negative + 1) % 3, minor = (negative + 2) % 3;
        if (q[major][major] > q[minor][minor]) { const int tmp = major; major = minor; minor = tmp; }
        const float lambdaA = q[major][major], lambdaB = q[minor][minor];
        if (!(lambdaA > 0.0f && lambdaB > 0.0f)) return false;
        // det(Q) is known analytically. This recovers the small negative
        // eigenvalue without cancellation for distant and grazing ellipses.
        const float minusLambda = (q00 / lambdaA) * (q11 / lambdaB) * invScale * invScale;
        if (!(minusLambda > 0.0f) || !isfinite(minusLambda)) return false;
        a = sqrtf(minusLambda / (lambdaA + minusLambda));
        b = sqrtf(minusLambda / (lambdaB + minusLambda));
        cosA2 = lambdaA / (lambdaA + minusLambda);
        cosB2 = lambdaB / (lambdaB + minusLambda);
        const float3 localX = make_float3(vectors[0][major], vectors[1][major], vectors[2][major]);
        const float3 localZ = make_float3(vectors[0][negative], vectors[1][negative], vectors[2][negative]);
        x = normalize(axisX * localX.x + axisY * localX.y + normal * localX.z);
        z = normalize(axisX * localZ.x + axisY * localZ.y + normal * localZ.z);
        if (dot(z, centerDelta) < 0.0f) z = -z;
        y = normalize(cross(z, x));
        x = cross(y, z);
        quadrantIntegral = integral(1.57079632679489661923f);
        solidAngle = 4.0f * a * b * quadrantIntegral;
        return isfinite(solidAngle) && solidAngle >= minSolidAngle;
    }

    SUTIL_HOSTDEVICE inline float lutAngle(float fraction) const {
        const float pa = fminf(15.0f, sqrtf(cosA2)*15.0f);
        const float pb = fminf(15.0f, sqrtf(cosB2)*15.0f);
        const float pu = fminf(32.0f, fraction*32.0f);
        const int ia = int(fminf(pa,14.0f)), ib = int(fminf(pb,14.0f));
        const int iu = int(fminf(pu,31.0f));
        const float fa = pa-ia, fb = pb-ib, fu = pu-iu;
        const int base = (ia*16+ib)*33+iu;
        const float aa = ellipseInverseCdf[base]*(1.0f-fu)+ellipseInverseCdf[base+1]*fu;
        const float ab = ellipseInverseCdf[base+33]*(1.0f-fu)+ellipseInverseCdf[base+34]*fu;
        const float ba = ellipseInverseCdf[base+528]*(1.0f-fu)+ellipseInverseCdf[base+529]*fu;
        const float bb = ellipseInverseCdf[base+561]*(1.0f-fu)+ellipseInverseCdf[base+562]*fu;
        return ((aa*(1.0f-fb)+ab*fb)*(1.0f-fa)+(ba*(1.0f-fb)+bb*fb)*fa)*1.5707963267948966f;
    }

    SUTIL_HOSTDEVICE inline float sectorAngle(float fraction, bool useLut,
                                             int* evaluations = nullptr) const {
        const float halfPi = 1.57079632679489661923f;
        const float target = fraction * quadrantIntegral;
        // The linear seed is already excellent for nearly circular projections;
        // avoid table traffic and interpolation noise in that limit.
        float t = useLut && fabsf(cosA2-cosB2)>1.0e-5f ? lutAngle(fraction) : fraction * halfPi;
        // Both paths use the same CDF, residual threshold, and five-step budget.
        for (int i = 0; i < 5; ++i) {
            if (evaluations) ++*evaluations;
            const float error = integral(t) - target;
            if (fabsf(error) <= 2.0e-7f * quadrantIntegral) break;
            t = fminf(halfPi, fmaxf(0.0f, t - error / density(t)));
        }
        return t;
    }

    SUTIL_HOSTDEVICE inline float3 sampleWithMethod(const float2& u, bool useLut) const {
        const float q = fminf(fmaxf(u.x, 0.0f), 0.99999994f) * 4.0f;
        const int quadrant = int(q);
        float fraction = q - float(quadrant);
        if (quadrant & 1) fraction = 1.0f - fraction;
        const float t = sectorAngle(fraction,useLut);
        const float ct = cosf(t), st = sinf(t);
        const float bx = a * ct, by = b * st;
        const float radius2 = bx * bx + by * by;
        const float boundaryZ = sqrtf(fmaxf(0.0f, cosA2 * ct * ct + cosB2 * st * st));
        const float height = fminf(fmaxf(u.y, 0.0f), 1.0f) * radius2 / (1.0f + boundaryZ);
        const float radialScale = sqrtf(fmaxf(0.0f, height * (2.0f - height) / radius2));
        const float sx = (quadrant == 1 || quadrant == 2) ? -1.0f : 1.0f;
        const float sy = quadrant >= 2 ? -1.0f : 1.0f;
        return normalize(x * (sx * bx * radialScale) + y * (sy * by * radialScale) + z * (1.0f - height));
    }

    SUTIL_HOSTDEVICE inline float3 sample(const float2& u) const {
        return sampleWithMethod(u,ZENO_ELLIPSE_USE_LUT != 0);
    }
};
