#pragma once 
#include <Sampling.h>
#include <LightBounds.h>
#include <sutil/vec_math.h>

#ifdef __CUDACC_RTC__
#include "zxxglslvec.h"
#else
#include "Host.h"
#endif

struct LightSampleRecord {
    float3 p;
    float PDF;

    float3 n;
    float NoL;

    float3 dir;
    float dist;

    float2 uv;

    float intensity = 1.0f;
    bool isDelta = false;
};

static constexpr float MinSphericalSampleArea = 3e-4f;
static constexpr float MaxSphericalSampleArea = 6.22f;

struct TriangleShape {
    float3 p0, p1, p2;
    float3 faceNormal;
    float  area;

    uint32_t coordsBufferOffset;
    uint32_t normalBufferOffset;

    inline float areaPDF() {
        return 1.0f / area;
    }
    
    float Area() const {
        return 0.5f * length(cross(p1 - p0, p2 - p0));
    }

    pbrt::Bounds3f bounds() {

        float3 pmax = make_float3(-FLT_MAX);
        float3 pmin = make_float3( FLT_MAX);

        float3 tmp[3] = {p0, p1, p2};

        for (int i=0; i<3; i++) {
            pmax = fmaxf(pmax, tmp[i]);
            pmin = fminf(pmin, tmp[i]);
        }

        pbrt::Bounds3f result;
        result.pMax = reinterpret_cast<Vector3f&>(pmax);
        result.pMin = reinterpret_cast<Vector3f&>(pmin);

        return result;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        auto& nnn = reinterpret_cast<Vector3f&>(faceNormal);
        auto dc = pbrt::DirectionCone(nnn);

        return pbrt::LightBounds(bounds(), nnn, phi * area, 
                dc.cosTheta, fmaxf(cosf(M_PIf / 2.0f), 0.0f), doubleSided);   
    }

    // Sampling Function Definitions
    float3 SampleSphericalTriangle(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, 
                                   const Vector3f& p, const float2& uu, float *pdf) {
        if (pdf)
            *pdf = 0;
        // Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
        Vector3f a(v0 - p), b(v1 - p), c(v2 - p);

        DCHECK(lengthSquared(a) > 0);
        DCHECK(lengthSquared(b) > 0);
        DCHECK(lengthSquared(c) > 0);
        a = normalize(a);
        b = normalize(b);
        c = normalize(c);

        // Compute normalized cross products of all direction pairs
        Vector3f n_ab = cross(a, b), n_bc = cross(b, c), n_ca = cross(c, a);
        if (lengthSquared(n_ab) == 0 || lengthSquared(n_bc) == 0 || lengthSquared(n_ca) == 0)
            return {};
        n_ab = normalize(n_ab);
        n_bc = normalize(n_bc);
        n_ca = normalize(n_ca);

        // Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
        float alpha = pbrt::AngleBetween(n_ab, -n_ca);
        float beta  = pbrt::AngleBetween(n_bc, -n_ab);
        float gamma = pbrt::AngleBetween(n_ca, -n_bc);

        // Uniformly sample triangle area $A$ to compute $A'$
        float A_pi = alpha + beta + gamma; // area + pi
        float Ap_pi = pbrt::Lerp(uu.x, M_PIf, A_pi);
        if (pdf) {
            float A = A_pi - M_PIf;
            *pdf = (A <= 0) ? 0 : 1 / A;
        }

        // Find $\cos\beta'$ for point along _b_ for sampled area
        float cosAlpha = cosf(alpha), sinAlpha = sinf(alpha);
        float sinPhi   = sinf(Ap_pi) * cosAlpha - cosf(Ap_pi) * sinAlpha;
        float cosPhi   = cosf(Ap_pi) * cosAlpha + sinf(Ap_pi) * sinAlpha;
        float k1 = cosPhi + cosAlpha;
        float k2 = sinPhi - sinAlpha * dot(a, b) /* cos c */;
        float cosBp = (k2 + (pbrt::DifferenceOfProducts(k2, cosPhi, k1, sinPhi)) * cosAlpha) /
                    ((pbrt::SumOfProducts(k2, sinPhi, k1, cosPhi)) * sinAlpha);
        // Happens if the triangle basically covers the entire hemisphere.
        // We currently depend on calling code to detect this case, which
        // is sort of ugly/unfortunate.
        DCHECK(!isnan(cosBp));
        cosBp = clamp(cosBp, -1.0f, 1.0f);

        // Sample $c'$ along the arc between $b'$ and $a$
        float sinBp = pbrt::SafeSqrt(1 - pbrt::Sqr(cosBp));
        Vector3f cp = cosBp * a + sinBp * normalize(pbrt::GramSchmidt(c, a));

        // Compute sampled spherical triangle direction and return barycentrics
        float cosTheta = 1 - uu.y * (1 - dot(cp, b));
        float sinTheta = pbrt::SafeSqrt(1 - pbrt::Sqr(cosTheta));
        Vector3f w = cosTheta * b + sinTheta * normalize(pbrt::GramSchmidt(cp, b));

        // Find barycentric coordinates for sampled direction _w_
        Vector3f e1 = v1 - v0, e2 = v2 - v0;
        Vector3f s1 = cross(w, e2);
        float divisor = dot(s1, e1);

        //CHECK_RARE(1e-6, divisor == 0);
        if (divisor == 0) {
            // This happens with triangles that cover (nearly) the whole
            // hemisphere.
            return {1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
        }
        float invDivisor = 1.0f / divisor;
        Vector3f s = p - v0;
        float b1 = dot(s, s1) * invDivisor;
        float b2 = dot(w, cross(s, e1)) * invDivisor;

        // Return clamped barycentrics for sampled direction
        b1 = clamp(b1, 0.0f, 1.0f);
        b2 = clamp(b2, 0.0f, 1.0f);
        if (b1 + b2 > 1) {
            b1 /= b1 + b2;
            b2 /= b1 + b2;
        }
        return {float(1 - b1 - b2), float(b1), float(b2)};
    }

    inline float SphericalTriangleArea(const float3& a, const float3& b, const float3& c) {
        return fabsf(2 * atan2f(dot(a, cross(b, c)), 1 + dot(a, b) + dot(a, c) + dot(b, c)));
    }

    inline float3 UniformSampleTriangle(const float2 &uu) {
        // float su0 = sqrtf(uu.x);
        // return make_float2(1 - su0, uu.y * su0);
        float b0, b1;
        if (uu.x < uu.y) {
            b0 = uu.x / 2;
            b1 = uu.y - b0;
        } else {
            b1 = uu.y / 2;
            b0 = uu.x - b1;
        }
        return {b0, b1, 1 - b0 - b1};
    }

    inline float SolidAngle(const float3& p) {

        auto v0 = normalize(p0 - p);
        auto v1 = normalize(p1 - p);
        auto v2 = normalize(p2 - p);

        return SphericalTriangleArea(v0, v1, v2);
    }

    inline void SampleAsLight(LightSampleRecord* lsr, float2& uu, const float3& shadingP, const float3& shadingN, 
                              const float3* vertexNormalBuffer, const float2* vertexCoordsBuffer) { 

        float solidAngle = SolidAngle(shadingP);
        if (solidAngle > MaxSphericalSampleArea || solidAngle < MinSphericalSampleArea) {

            auto bary3 = UniformSampleTriangle(uu);
            lsr->p = bary3.x * p0 + bary3.y * p1 + bary3.z * p2;
            lsr->n = faceNormal;

            lsr->dir = (lsr->p - shadingP);
            //lsr->dir = normalize(lsr->dir);
            auto sign = copysignf(1.0f, dot(lsr->n, -lsr->dir));

            lsr->p = rtgems::offset_ray(lsr->p, lsr->n * sign);
            lsr->dir = lsr->p - shadingP;
            lsr->dist = length(lsr->dir);

            if (lsr->dist == 0.0f) return;
            lsr->dir = lsr->dir / lsr->dist;

            if (coordsBufferOffset != UINT_MAX && vertexCoordsBuffer != nullptr) { // has vertex UV coord
                auto& uv0 = vertexCoordsBuffer[coordsBufferOffset + 0];
                auto& uv1 = vertexCoordsBuffer[coordsBufferOffset + 1];
                auto& uv2 = vertexCoordsBuffer[coordsBufferOffset + 2];

                lsr->uv = bary3.x * uv0 + bary3.y * uv1 + bary3.z * uv2;
            }

            if (normalBufferOffset != UINT_MAX && vertexNormalBuffer != nullptr) { // has vertex normal
                auto& vn0 = vertexNormalBuffer[normalBufferOffset + 0];
                auto& vn1 = vertexNormalBuffer[normalBufferOffset + 1];
                auto& vn2 = vertexNormalBuffer[normalBufferOffset + 2];

                lsr->n = bary3.x * vn0 + bary3.y * vn1 + bary3.z * vn2;
            } 

            lsr->NoL = dot(-lsr->dir, lsr->n);
            lsr->PDF = 0.0f;

            if (fabsf(lsr->NoL) > __FLT_EPSILON__) {
                lsr->PDF = lsr->dist * lsr->dist * areaPDF() / fabsf(lsr->NoL);
            }
        } // uniform area sampling

        float pdf = 1.0f;

        if (dot(shadingN, shadingN) > 0) { // shading point not in volume

            float3 wi[3] = { normalize(p0 - shadingP), 
                             normalize(p1 - shadingP), 
                             normalize(p2 - shadingP) };

            float4 v = {   fmaxf(0.01, fabsf(dot(shadingN, wi[1]))),
                           fmaxf(0.01, fabsf(dot(shadingN, wi[1]))),
                           fmaxf(0.01, fabsf(dot(shadingN, wi[0]))),
                           fmaxf(0.01, fabsf(dot(shadingN, wi[2])))
                        };
            uu = pbrt::SampleBilinear(uu, v);
            DCHECK(uu[0] >= 0 && uu[0] < 1);
            DCHECK(uu[1] >= 0 && uu[1] < 1);
            pdf = pbrt::BilinearPDF(uu, v);
        }

        float triPDF;

        float3 bary3 = SampleSphericalTriangle(reinterpret_cast<Vector3f&>(p0),
                                               reinterpret_cast<Vector3f&>(p1),
                                               reinterpret_cast<Vector3f&>(p2), 
                                               reinterpret_cast<const Vector3f&>(shadingP), uu, &triPDF);
        if (triPDF == 0) { return; }
        pdf *= triPDF;

        float3 p = bary3.x * p0 + bary3.y * p1 + bary3.z * p2;
        // Compute surface normal for sampled point on triangle
        float3 n = faceNormal; //normalize( cross(p1 - p0, p2 - p0) );

        lsr->p = p;
        lsr->n = n;

        lsr->dir = (lsr->p - shadingP);
        //lsr->dir = normalize(lsr->dir);
        auto sign = copysignf(1.0f, dot(lsr->n, -lsr->dir));

        lsr->p = rtgems::offset_ray(lsr->p, lsr->n * sign);
        lsr->dir = lsr->p - shadingP;
        lsr->dist = length(lsr->dir);
        lsr->dir = lsr->dir / lsr->dist;

        if (coordsBufferOffset != UINT_MAX && vertexCoordsBuffer != nullptr) { // has vertex UV coord
            auto& uv0 = vertexCoordsBuffer[coordsBufferOffset + 0];
            auto& uv1 = vertexCoordsBuffer[coordsBufferOffset + 1];
            auto& uv2 = vertexCoordsBuffer[coordsBufferOffset + 2];

            lsr->uv = bary3.x * uv0 + bary3.y * uv1 + bary3.z * uv2;
        }

        if (normalBufferOffset != UINT_MAX && vertexNormalBuffer != nullptr) { // has vertex normal
            auto& vn0 = vertexNormalBuffer[normalBufferOffset + 0];
            auto& vn1 = vertexNormalBuffer[normalBufferOffset + 1];
            auto& vn2 = vertexNormalBuffer[normalBufferOffset + 2];

            lsr->n = bary3.x * vn0 + bary3.y * vn1 + bary3.z * vn2;
        }

        lsr->NoL = dot(-lsr->dir, lsr->n);
        lsr->PDF = pdf;
    }

    // Via Jim Arvo's SphTri.C
    float2 InvertSphericalTriangleSample(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2,
                                         const Vector3f& p, const Vector3f& w) {
        // Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
        Vector3f a(v0 - p), b(v1 - p), c(v2 - p);
        DCHECK(LengthSquared(a) > 0);
        DCHECK(LengthSquared(b) > 0);
        DCHECK(LengthSquared(c) > 0);
        a = normalize(a);
        b = normalize(b);
        c = normalize(c);

        // Compute normalized cross products of all direction pairs
        Vector3f n_ab = cross(a, b), n_bc = cross(b, c), n_ca = cross(c, a);
        if (lengthSquared(n_ab) == 0 || lengthSquared(n_bc) == 0 || lengthSquared(n_ca) == 0)
            return {};
        n_ab = normalize(n_ab);
        n_bc = normalize(n_bc);
        n_ca = normalize(n_ca);

        // Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
        float alpha = pbrt::AngleBetween(n_ab, -n_ca);
        float beta  = pbrt::AngleBetween(n_bc, -n_ab);
        float gamma = pbrt::AngleBetween(n_ca, -n_bc);

        // Find vertex $\VEC{c'}$ along $\VEC{a}\VEC{c}$ arc for $\w{}$
        Vector3f cp = normalize(cross(cross(b, w), cross(c, a)));
        if (dot(cp, a + c) < 0)
            cp = -cp;

        // Invert uniform area sampling to find _u0_
        float u0;
        if (dot(a, cp) > 0.99999847691f /* 0.1 degrees */)
            u0 = 0;
        else {
            // Compute area $A'$ of subtriangle
            Vector3f n_cpb = cross(cp, b), n_acp = cross(a, cp);
            //CHECK_RARE(1e-5, lengthSquared(n_cpb) == 0 || lengthSquared(n_acp) == 0);
            
            if (lengthSquared(n_cpb) == 0 || lengthSquared(n_acp) == 0)
                return make_float2(0.5, 0.5);

            n_cpb = normalize(n_cpb);
            n_acp = normalize(n_acp);
            float Ap = alpha + pbrt::AngleBetween(n_ab, n_cpb) + pbrt::AngleBetween(n_acp, -n_cpb) - M_PIf;

            // Compute sample _u0_ that gives the area $A'$
            float A = alpha + beta + gamma - M_PIf;
            u0 = Ap / A;
        }

        // Invert arc sampling to find _u1_ and return result
        float u1 = (1 - dot(w, b)) / (1 - dot(cp, b));
        return make_float2(clamp(u0, 0.f, 1.f), clamp(u1, 0.f, 1.f));
    }

    inline void EvalAfterHit(LightSampleRecord* lsr, const float3& dir, const float& dist, 
                            const float3& shadingP, const float3& shadingN, const float3& barys3, 
                            const float3* vertexNormalBuffer, const float2* vertexCoordsBuffer) {

        lsr->p = barys3.x * p0 + barys3.y * p1 + barys3.z * p2;

        lsr->n = faceNormal;
        if (normalBufferOffset != UINT_MAX && vertexNormalBuffer != nullptr) { // has vertex normal
            auto& vn0 = vertexNormalBuffer[normalBufferOffset + 0];
            auto& vn1 = vertexNormalBuffer[normalBufferOffset + 1];
            auto& vn2 = vertexNormalBuffer[normalBufferOffset + 2];

            lsr->n = barys3.x * vn0 + barys3.y * vn1 + barys3.z * vn2;
        }

        if (coordsBufferOffset != UINT_MAX && vertexCoordsBuffer != nullptr) { // has vertex UV coord
            auto& uv0 = vertexCoordsBuffer[coordsBufferOffset + 0];
            auto& uv1 = vertexCoordsBuffer[coordsBufferOffset + 1];
            auto& uv2 = vertexCoordsBuffer[coordsBufferOffset + 2];

            lsr->uv = barys3.x * uv0 + barys3.y * uv1 + barys3.z * uv2;
        }

        float solidAngle = SolidAngle(shadingP);
        if (solidAngle > MaxSphericalSampleArea || solidAngle < MinSphericalSampleArea) {
            // Compute PDF in solid angle measure from shape intersection point

            float lightNoL = dot(-dir, lsr->n);
            float lightPDF = ( dist * dist / Area()) / fabsf(lightNoL);

            if (isinf(lightPDF) || isnan(lightPDF))
                lightPDF = 0;

            lsr->dir = dir;
            lsr->dist = dist;
            lsr->NoL = lightNoL;
            lsr->PDF = lightPDF;
            return;   
        }

        float pdf = 1.0f / solidAngle;
        // Adjust PDF for warp product sampling of triangle $\cos\theta$ factor
        if (dot(shadingN, shadingN) > 0.0f) { // not in volume

            float2 uu = InvertSphericalTriangleSample(reinterpret_cast<const Vector3f&>(p0),
                                                      reinterpret_cast<const Vector3f&>(p1), 
                                                      reinterpret_cast<const Vector3f&>(p2), 
                                                      reinterpret_cast<const Vector3f&>(shadingP), 
                                                      reinterpret_cast<const Vector3f&>(dir));
                                                     
            float3 vvv[3] { 
                normalize(p0 - shadingP), 
                normalize(p1 - shadingP), 
                normalize(p2 - shadingP) 
            };

            float4 w { 
                fmaxf(0.01, pbrt::AbsDot(shadingN, vvv[1])),
                fmaxf(0.01, pbrt::AbsDot(shadingN, vvv[1])),
                fmaxf(0.01, pbrt::AbsDot(shadingN, vvv[0])),
                fmaxf(0.01, pbrt::AbsDot(shadingN, vvv[2]))
            };

            pdf *= pbrt::BilinearPDF(uu, w);
        }

        lsr->dir = dir;
        lsr->dist = dist;
        lsr->PDF = pdf;
        lsr->NoL = dot(-dir, lsr->n);

        return;
    } 
};

struct PointShape {
    float3 p;

    inline float PDF() {return 0.25f / M_PIf;}

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {

        auto vector = p - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->dist = dist;
        lsr->dir = vector / dist;
        lsr->p = p;
        lsr->n = -lsr->dir;

        lsr->PDF = 1.0f; //PDF();
        lsr->NoL = 1.0f;
        lsr->intensity = M_PIf / dist2;
        lsr->isDelta = true;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        float Phi = 4 * M_PIf * phi; 

        auto& tmp = reinterpret_cast<Vector3f&>(p);
        auto bounds = pbrt::Bounds3f{tmp, tmp};
        
        return pbrt::LightBounds(bounds, Vector3f(0, 0, 1), 
            Phi, cosf(M_PIf), cosf(M_PIf / 2), false);
    }
};

struct SphericalRect {
    float3 o, x, y, z;
    float z0, z0sq;
    float x0, y0, y0sq;
    float x1, y1, y1sq;
    float b0, b1, b0sq, k;
    float S;
};

__device__ inline void SphericalRectInit(SphericalRect& srect, 
    const float3& o, const float3& s, 
    const float3& axisX, const float& lenX, 
    const float3& axisY, const float& lenY) {

    srect.o = o; 
    
    float exl = lenX, eyl = lenY;
    // compute local reference system ’R’
    srect.x = axisX;
    srect.y = axisY;
    srect.z = cross(srect.x, srect.y);
    // compute rectangle coords in local reference system
    float3 d = s - o;
    srect.z0 = dot(d, srect.z);
    // flip ’z’ to make it point against ’Q’
    if (srect.z0 > 0) {
        srect.z *= -1.f;
        srect.z0 *= -1.f;
    }
    srect.z0sq = srect.z0 * srect.z0;
    srect.x0 = dot(d, srect.x);
    srect.y0 = dot(d, srect.y);
    srect.x1 = srect.x0 + exl;
    srect.y1 = srect.y0 + eyl;
    srect.y0sq = srect.y0 * srect.y0;
    srect.y1sq = srect.y1 * srect.y1;
    // create vectors to four vertices
    float3 v00 = {srect.x0, srect.y0, srect.z0};
    float3 v01 = {srect.x0, srect.y1, srect.z0};
    float3 v10 = {srect.x1, srect.y0, srect.z0};
    float3 v11 = {srect.x1, srect.y1, srect.z0};
    // compute normals to edges
    float3 n0 = normalize(cross(v00, v10));
    float3 n1 = normalize(cross(v10, v11));
    float3 n2 = normalize(cross(v11, v01));
    float3 n3 = normalize(cross(v01, v00));
    // compute internal angles (gamma_i)
    float g0 = acosf(-dot(n0,n1));
    float g1 = acosf(-dot(n1,n2));
    float g2 = acosf(-dot(n2,n3));
    float g3 = acosf(-dot(n3,n0));
    // compute predefined constants
    srect.b0 = n0.z;
    srect.b1 = n2.z;
    srect.b0sq = srect.b0 * srect.b0;
    srect.k = 2.0f * M_PIf - g2 - g3;
    // compute solid angle from internal angles
    srect.S = g0 + g1 - srect.k;
}

static inline float2 SphericalRectSample(SphericalRect& srect, float u, float v) {
    // 1. compute ’cu’
    float au = u * srect.S + srect.k;
    if(abs(sinf(au))<1e-5)
    {
      return {0, 0};
    }
    float fu = (cosf(au) * srect.b0 - srect.b1) / sinf(au) ;
    float cu = (fu>0 ? +1.f:-1.f) /sqrtf(fu*fu + srect.b0sq);
    cu = clamp(cu, -1.0f, 1.0f); // avoid NaNs
    // 2. compute ’xu’
    float xu = -(cu * srect.z0) / sqrtf(1.f - cu*cu);
    xu = clamp(xu, srect.x0, srect.x1); // avoid Infs
    // 3. compute ’yv’
    float d = sqrtf(xu*xu + srect.z0sq);
    float h0 = srect.y0 / sqrtf(d*d + srect.y0sq);
    float h1 = srect.y1 / sqrtf(d*d + srect.y1sq);
    float hv = h0 + v * (h1-h0), hv2 = hv*hv;
    float yv = (hv2 < 1.f-__FLT_EPSILON__) ? (hv*d)/sqrtf(1.f-hv2) : srect.y1;
    // 4. transform (xu,yv,z0) to world coords
    //return (squad.o + xu*squad.x + yv*squad.y + squad.z0*squad.z);

    return { (xu-srect.x0) / (srect.x1 - srect.x0),
             (yv-srect.y0) / (srect.y1 - srect.y0) };
}

__device__ inline bool SpreadClampRect(float3& v,
                    const float3& axisX, float& lenX, 
                    const float3& axisY, float& lenY,
                    const float3& normal, const float3& shadingP, 
                    float spread, float2& uvScale, float2& uvOffset, bool isEllipse=false) {

    if (spread >= 1.0f) {
        uvScale  = {1.0f, 1.0f};
        uvOffset = {0.0f, 0.0f}; 
        return true;
    }

    auto diff = shadingP - v;
    
    auto vDistance = fabsf(dot(diff, normal));
    auto spread_angle = 0.5f * spread * M_PIf;
    auto spread_radius = tanf(spread_angle) * vDistance;

    auto posU = dot(diff, axisX);
    auto posV = dot(diff, axisY);

    // if (posU > lenU + spread_radius || posU < -spread_radius)
    //     return false;
    // if (posV > lenV + spread_radius || posV < -spread_radius)
    //     return false;

    auto minU = fmaxf(posU-spread_radius, 0.0f);
    auto maxU = fminf(posU+spread_radius, lenX);

    auto minV = fmaxf(posV-spread_radius, 0.0f);
    auto maxV = fminf(posV+spread_radius, lenY);

    if (minU > maxU || minV > maxV) return false;

    auto lenU = maxU - minU;
    auto lenV = maxV - minV;

    uvScale  = {lenU/lenX, lenV/lenY};
    uvOffset = {minU/lenX, minV/lenY};

    v = v + minU * axisX + minV *axisY;

    if (isEllipse) {

        float2 uvCenter {0.5f, 0.5f};

        float2 subCenter {
            uvOffset.x + uvScale.x * 0.5f,
            uvOffset.y + uvScale.y * 0.5f };

        auto delta1 = uvCenter - subCenter;

        auto reX1 = copysignf(1.0f, delta1.x);
        auto reY1 = copysignf(1.0f, delta1.y);

        float2 subCorner {
            subCenter.x + reX1 * uvScale.x * 0.5f,
            subCenter.y + reY1 * uvScale.x * 0.5f };

        auto delta2 = uvCenter - subCorner;

        auto reX2 = copysignf(1.0f, delta2.x);
        auto reY2 = copysignf(1.0f, delta2.y);
        
        if (reX2 == reX1 && reY2 == reY1) { // signbit()

            auto lengthUV = length(subCorner - uvCenter);
            if (lengthUV > 0.5f) { return false; }
        }
    }
    
    lenX = lenU;
    lenY = lenV;

    return true;
}

struct RectShape {
    float3 v; bool isEllipse:1;
    float3 axisX; float lenX;
    float3 axisY; float lenY;

    float3 normal;
    float  area;

    float Area() {
        if (isEllipse)
            return 0.25 * M_PIf * lenX * lenY;
        else 
            return lenX * lenY;
    }

    inline float PDF() {
        return 1.0f / Area();
    }

    inline bool EvalAfterHit(LightSampleRecord* lsr, const float3& dir, const float& dist, const float3& shadingP) {

        float lightNoL = dot(-dir, normal);
        float lightPDF = dist * dist * PDF() / lightNoL;
        
        lsr->p = shadingP + dir * dist;
        lsr->n = normal;
        lsr->dir = dir;
        lsr->dist = dist;

        auto delta = lsr->p - v;
        delta -= dot(delta, normal) * normal;
            
        lsr->uv = { dot(delta, axisX) / lenX,
                    dot(delta, axisY) / lenY };

        if (isEllipse) {
            auto uvd = lsr->uv - 0.5f;
            if (length(uvd) > 0.5f) { return false; }
        }

        lsr->PDF = lightPDF;
        lsr->NoL = lightNoL;

        return true;
    }  

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {  

        auto uv = uu; 
        auto _PDF_ = 0.0f;

        if (isEllipse) {

            auto tt = pbrt::SampleUniformDiskConcentric(uu);
            tt = tt * 0.5f + 0.5f;
            uv = tt;
            _PDF_ = PDF();

        } else {

            SphericalRect squad;
            SphericalRectInit(squad, shadingP, v, axisX, lenX, axisY, lenY); 
            uv = SphericalRectSample(squad, uu.x, uu.y);
            _PDF_ = squad.S;
        }

        lsr->n = normalize(normal);
        lsr->p = v + axisX * lenX * uv.x + axisY * lenY * uv.y;

        lsr->uv = uv;

        lsr->dir = (lsr->p - shadingP);
        //lsr->dir = normalize(lsr->dir);
        auto sign = copysignf(1.0f, dot(lsr->n, -lsr->dir));
        
        lsr->p = rtgems::offset_ray(lsr->p, lsr->n * sign);
        lsr->dir = lsr->p - shadingP;
        lsr->dist = length(lsr->dir);
        lsr->dir = lsr->dir / lsr->dist;

        lsr->NoL = dot(-lsr->dir, lsr->n);
        lsr->PDF = 0.0f;

        if (_PDF_ > __FLT_EPSILON__ && fabsf(lsr->NoL) > __FLT_EPSILON__) 
        {
            if (isEllipse) {
                lsr->PDF = lsr->dist * lsr->dist * _PDF_ / fabsf(lsr->NoL);
            } else {
                lsr->PDF = 1.0f / _PDF_;
            }
        }
    }

    inline bool hitAsLight(LightSampleRecord* lsr, const float3& ray_orig, const float3& ray_dir) {

        // assuming normal and ray_dir are normalized
        float denom = dot(normal, -ray_dir);
        if (denom <= __FLT_DENORM_MIN__) {return false;}
        
        float3 vector = ray_orig - v;
        float t = dot(normal, vector) / denom;

        if (t <= 0) { return false; }

        auto P = ray_orig + ray_dir * t;
        auto delta = P - v;

        P -= normal * dot(normal, delta);
        delta = P - v; 
        
        auto q1 = dot(delta, axisX);
        if (q1<0.0f || q1>lenX) {return false;}
       
        auto q2 = dot(delta, axisY);
        if (q2<0.0f || q2>lenY) {return false;}

        lsr->uv = float2{q1, q2} / float2{lenX, lenY};

        if (isEllipse) {
            auto uvd = lsr->uv - 0.5f;
            if (length(uvd) > 0.5f) { return false; }
        }

        lsr->p = P;
        lsr->PDF = 1.0f;
        lsr->n = normal;
        lsr->NoL = denom;

        lsr->p = rtgems::offset_ray(lsr->p, lsr->n);

        lsr->dir = ray_dir;
        lsr->dist = length(lsr->p - ray_orig);

        return true;
    }

    pbrt::Bounds3f bounds() {

        auto pmax = v;
        auto pmin = v;

        auto v0 = v;
        auto v1 = axisX * lenX;
        auto v2 = axisY * lenY;

        float3 tmp[3] = {v0+v1, v0+v2, v0+v1+v2};

        for (int i=0; i<3; i++) {
            pmax = fmaxf(pmax, tmp[i]);
            pmin = fminf(pmin, tmp[i]);
        }

        pbrt::Bounds3f result;
        result.pMax = reinterpret_cast<Vector3f&>(pmax);
        result.pMin = reinterpret_cast<Vector3f&>(pmin);

        return result;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        auto& nnn = reinterpret_cast<Vector3f&>(normal);
        auto dc = pbrt::DirectionCone(nnn);

        return pbrt::LightBounds(bounds(), nnn, phi * area, 
                dc.cosTheta, fmaxf(cosf(M_PIf / 2.0f), 0.0f), doubleSided);   
    }
};

struct ConeShape {
    float3 p;
    float range;

    float3 dir;
    float cosFalloffStart;
    float cosFalloffEnd;

    inline void sample(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {
        auto vector = p - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->dir = vector / dist;
        lsr->dist = dist;

        lsr->n = dir;
        lsr->NoL = dot(-lsr->dir, dir);

        lsr->p = p;
        lsr->PDF = 1.0f;

        #ifdef __CUDACC_RTC__
        lsr->intensity = smoothstep(cosFalloffEnd, cosFalloffStart, lsr->NoL);
        #endif

        lsr->intensity *= M_PIf / dist2;
    }

    inline float Phi() {
        return 2 * M_PIf * ((1.0f - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2.0f);
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        auto Phi = M_PIf * 4 * phi;

        float cosTheta_e = cosf(acosf(cosFalloffEnd) - acosf(cosFalloffStart));
        // Allow a little slop here to deal with fp round-off error in the computation of
        // cosTheta_p in the importance function.
        if (cosTheta_e == 1 && cosFalloffEnd != cosFalloffStart)
            cosTheta_e = 0.999f;

        auto& w = reinterpret_cast<Vector3f&>(dir);
        auto& tmp = reinterpret_cast<Vector3f&>(p);
        auto bounds = pbrt::Bounds3f{tmp, tmp};

        return pbrt::LightBounds(bounds, w, Phi, cosFalloffStart, cosTheta_e, false);
    }
};

struct SphereShape {
    float3 center;
    float  radius;
    float  area;

    inline float PDF() {
        return 1.0f / area;
    }

    inline float PDF(const float3& shadingP, float dist2, float NoL) {

        if (dist2 < radius * radius) {
            return dist2 / fabsf(NoL);
        }

        float sinThetaMax2 = clamp( radius * radius / dist2, 0.0, 1.0);

        if (sinThetaMax2 <= __FLT_EPSILON__) {
            return 1.0f; // point light
        }

        float cosThetaMax = sqrtf( 1.0 - sinThetaMax2 );
        return 1.0f / ( 2.0f * M_PIf * (1.0 - cosThetaMax) );
    }

    inline bool hitAsLight(LightSampleRecord* lsr, const float3& ray_origin, const float3& ray_dir) {

        float3 f = ray_origin - center;
        float b2 = dot(f, ray_dir);
        if (b2 >= 0) { return false; }

		float r2 = radius * radius;

		float3 fd = f - b2 * ray_dir;
		float discriminant = r2 - dot(fd, fd);

		if (discriminant >= 0.0f)
		{
			float c = dot(f, f) - r2;
			float sqrtVal = sqrt(discriminant);

			// include Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery, 
			// "Numerical Recipes in C," Cambridge University Press, 1992.
			float q = (b2 >= 0) ? -sqrtVal - b2 : sqrtVal - b2;

            lsr->dir = ray_dir;
            lsr->dist = fminf(c/q, q);
            lsr->p = ray_origin + ray_dir * lsr->dist;
            lsr->n = normalize(lsr->p - center);
            lsr->p = center + radius * lsr->n;
            lsr->p = rtgems::offset_ray(lsr->p, lsr->n);
            lsr->dist = length(lsr->p - ray_origin);
            return true;

			// we don't bother testing for division by zero
			//ReportHit(c / q, 0, sphrAttr);
			// more distant hit - not needed if we know we will intersect with the outside of the sphere
			//ReportHit(q / a, 0, sphrAttr);
		}
        return false;
    }

    inline void EvalAfterHit(LightSampleRecord* lsr, const float3& dir, const float& distance, const float3& shadingP) {

        auto vector = center - shadingP;
        auto dist2 = dot(vector, vector);
        auto dist = sqrtf(dist2);

        lsr->p = shadingP + dir * distance;
        lsr->n = normalize(lsr->p - center);
        if (dist2 < radius * radius) {
            lsr->n *= -1;
        }

        lsr->NoL = dot(lsr->n, -dir);
        lsr->PDF = PDF(shadingP, dist2, lsr->NoL);

        lsr->dir = dir;
        lsr->dist = distance;
    }

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) {

        float3 vector = center - shadingP;
        float  dist2 = dot(vector, vector);
        float  dist = sqrtf(dist2);
        float3 dir = vector / dist;

        float radius2 = radius * radius;

        if (dist2 <= radius2) { // inside sphere
            lsr->PDF = 0.0f;
            return;
            
            auto localP = pbrt::UniformSampleSphere(uu);
            auto worldP = center + localP * radius;

            auto localN = -localP; //facing center
            auto worldN =  localN; 

            lsr->p = rtgems::offset_ray(worldP, worldN);
            lsr->n = worldN;

            vector = lsr->p - shadingP;
            dist2 = dot(vector, vector);

            if (dist2 == 0) {
                lsr->PDF = 0.0f; return;
            }

            dist = sqrtf(dist2);
            dir = vector / dist;

            lsr->dist = dist;
            lsr->dir  = dir;

            lsr->NoL = dot(-dir, worldN);
            lsr->PDF = lsr->dist * lsr->dist / lsr->NoL;
            return;       
        }

        // Sample sphere uniformly inside subtended cone
        float invDc = 1.0f / dist;
        float3& wc = dir; float3 wcX, wcY;
        pbrt::CoordinateSystem(wc, wcX, wcY);

        // Compute $\theta$ and $\phi$ values for sample in cone
        float sinThetaMax = radius * invDc;
        const float sinThetaMax2 = sinThetaMax * sinThetaMax;
        float invSinThetaMax = 1.0f / sinThetaMax;

        assert(sinThetaMax2 > 0);
        const float cosThetaMax = sqrtf(1.0f - clamp(sinThetaMax2, 0.0f, 1.0f));

        auto epsilon = 2e-3f;

        if (sinThetaMax < epsilon) {
            
            lsr->p = center - dir * radius;
            lsr->p = rtgems::offset_ray(lsr->p, -dir);

            lsr->n = -dir;
            lsr->dir = dir;
            lsr->dist = length(lsr->p - shadingP);

            lsr->PDF = 1.0f;
            lsr->NoL = 1.0f;
            lsr->intensity = M_PIf * radius2 / (lsr->dist * lsr->dist);
            lsr->isDelta = true;
            return;
        } // point light

        float cosTheta  = (cosThetaMax - 1) * uu.x + 1;
        float sinTheta2 = 1 - cosTheta * cosTheta;

        if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */) {
            /* Fall back to a Taylor series expansion for small angles, where
            the standard approach suffers from severe cancellation errors */
            sinTheta2 = sinThetaMax2 * uu.x;
            cosTheta = sqrtf(1 - sinTheta2);
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        float cosAlpha = sinTheta2 * invSinThetaMax +
              cosTheta * sqrtf(fmaxf(0.f, 1.f - sinTheta2 * invSinThetaMax * invSinThetaMax));
        float sinAlpha = sqrtf(fmaxf(0.f, 1.f - cosAlpha * cosAlpha));
        float phi = uu.y * 2 * M_PIf;

        // Compute surface normal and sampled point on sphere
        float3 nWorld = pbrt::SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
        float3 pWorld = center + radius * nWorld;

        lsr->p = rtgems::offset_ray(pWorld, nWorld);
        lsr->n = nWorld;

        vector = lsr->p - shadingP;
        dist2 = dot(vector, vector);
        dist = sqrtf(dist2);
        dir = vector / dist;

        lsr->dist = dist;
        lsr->dir  = dir;
        
        lsr->PDF = 1.0f / (2.0f * M_PIf * (1.0f - cosThetaMax)); // Uniform cone PDF.
        lsr->NoL = dot(-lsr->dir, lsr->n); 
    }

    pbrt::Bounds3f bounds() {

        auto pmax = center + make_float3(abs(radius));
        auto pmin = center - make_float3(abs(radius));

        pbrt::Bounds3f result;
        result.pMax = reinterpret_cast<Vector3f&>(pmax);
        result.pMin = reinterpret_cast<Vector3f&>(pmin);

        return result;
    }

    pbrt::LightBounds BoundAsLight(float phi, bool doubleSided) {

        auto dc = pbrt::DirectionCone::EntireSphere();

        return pbrt::LightBounds(bounds(), dc.w, phi * area, 
            dc.cosTheta, fmaxf(cos(M_PIf / 2.0f), 0.0f), doubleSided);   
    }
};
