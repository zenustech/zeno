#pragma once

#include <Sampling.h>

#ifndef __CUDACC_RTC__
    #include <zeno/utils/vec.h>

    #include <glm/common.hpp>
    #include <glm/gtx/transform.hpp>
    #include <glm/ext/matrix_transform.hpp>
#endif

#ifndef INFINITY
#define INFINITY CUDART_INF_F 
#endif

#ifndef FLT_MAX 
#define FLT_MAX __FLT_MAX__
#endif

#ifdef __CUDACC_RTC__

namespace zeno {

    inline Vector3f min(Vector3f a, Vector3f b) {
        return fminf(a, b);
    }

    inline Vector3f max(Vector3f a, Vector3f b) {
        return fmaxf(a, b);
    }
}

#endif

namespace pbrt {

struct Bounds3f {

#ifndef __CUDACC_RTC__
    using Vec3 = zeno::vec<3, float>;
#else
    using Vec3 = vec3;
#endif

    Vec3 pMin = Vec3 {FLT_MAX, FLT_MAX, FLT_MAX};
    Vec3 pMax = -Vec3 {FLT_MAX, FLT_MAX, FLT_MAX};

    Bounds3f() = default;
    // Bounds3f(Vec3 v) : pMin(v), pMax(v) {}
    // Bounds3f(Vec3 a, Vec3 b) {

    //     Vector3f x, y;

    //     for (int i=0; i<3; ++i) {
    //         x[i] = fminf(a[i], b[i]);
    //         y[i] = fmaxf(a[i], b[i]);
    //     }

    //     pMin = x;
    //     pMax = y;
    // } 

    Vec3 operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    
    Vec3 &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }

    Vec3 diagonal() const {
        return pMax - pMin;
    }

    Vec3 center() const {
        return (pMin + pMax) / 2;
    }

    Vec3 offset(Vec3 p) const {
        Vector3f _o_ = p - pMin;
        _o_ = _o_ / diagonal();
        return _o_;
    }

    float area() const {
        auto d = diagonal();
        return 2.0f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
    }

    float volume() const {
        auto d = diagonal();
        return d[0] * d[1] * d[2];
    }

    int maxDimension() const {
        auto d = diagonal();
        if (d[0] > d[1] && d[0] > d[2])
            return 0;
        else if (d[1] > d[2])
            return 1;
        else
            return 2;
    }

    bool inside(const Vec3& point) const {

        for (uint32_t i=0; i<3; ++i) {
            if (point[i] < pMin[i] || point[i] > pMax[i])
                return false;
        }
        return true;
    }

    void BoundingSphere(Vec3 *center, float *radius) const {
        *center = (pMin + pMax) / 2;
        *radius = inside(*center) ? distance(*center, pMax) : 0;
    }

    bool IsEmpty() const {
        return pMin[0] >= pMax[0] || pMin[1] >= pMax[1] || pMin[2] >= pMax[2];
    }
    
    bool IsDegenerate() const {
        return pMin[0] > pMax[0] || pMin[1] > pMax[1] || pMin[2] > pMax[2];
    }
};

inline float MaxComponentValue(const Vector3f vvv) {
    return fmaxf(fmaxf(vvv[0], vvv[1]), vvv[2]);
}

static Bounds3f Union(const Bounds3f &a, const Bounds3f &b) {

    Bounds3f result;

    result.pMin = zeno::min(a.pMin, b.pMin);
    result.pMax = zeno::max(a.pMax, b.pMax);

    return result;
}

static Bounds3f Union(const Bounds3f &a, const Vector3f &b) {

    Bounds3f result;

    result.pMin = zeno::min(a.pMin, b);
    result.pMax = zeno::max(a.pMax, b);

    return result;
}

static Bounds3f Union(const Vector3f &b, const Bounds3f &a) {
    return Union(a, b);
}

struct DirectionCone {

    Vector3f w;
    float cosTheta = INFINITY;

    DirectionCone() = default;

    explicit DirectionCone(Vector3f w) : DirectionCone(w, 1) {}

    DirectionCone(Vector3f w, float cosTheta) : w(normalize(w)), cosTheta(cosTheta) {}
     
    bool IsEmpty() const { return cosTheta == INFINITY; }

    static DirectionCone EntireSphere() { return DirectionCone(Vector3f(0, 0, 1), -1); }
};

// LightBounds Definition
struct LightBounds {

    Bounds3f bounds; Vector3f w{};

    float cosTheta_o{}, cosTheta_e{};
    float phi = 0.0f; bool doubleSided = false;

    LightBounds() = default;
    LightBounds(const Bounds3f &b, Vector3f w, float phi, float cosTheta_o, float cosTheta_e, bool doubleSided) {
        this->bounds = b;
        this->w = normalize(w); this->phi = phi;
        this->cosTheta_o = cosTheta_o;
        this->cosTheta_e = cosTheta_e;
        this->doubleSided = doubleSided;
    } 

    Vector3f centroid() const { return (bounds.pMin + bounds.pMax) / 2.0f; }

    float importance(Vector3f p, Vector3f n) const;
};

inline DirectionCone BoundSubtendedDirections(const Bounds3f &b, Vector3f p) {
    // Compute bounding sphere for _b_ and check if _p_ is inside
    float radius; Vector3f pCenter;
    b.BoundingSphere(&pCenter, &radius);

    float lenSquared = lengthSquared(pCenter - p);

    if (lenSquared < Sqr(radius))
        return DirectionCone::EntireSphere();

    // Compute and return _DirectionCone_ for bounding sphere
    Vector3f w = normalize(pCenter - p);
    float sin2ThetaMax = Sqr(radius) / lenSquared;
    float cosThetaMax = SafeSqrt(1 - sin2ThetaMax);
    return DirectionCone(w, cosThetaMax);
}

#ifndef __CUDACC_RTC__

inline void Inverse(DirectionCone& dc) {
    dc.w *= -1.0f;
}

// DirectionCone Function Definitions
static DirectionCone Union(const DirectionCone &a, const DirectionCone &b) {
    // Handle the cases where one or both cones are empty
    if (a.IsEmpty())
        return b;
    if (b.IsEmpty())
        return a;

    // Handle the cases where one cone is inside the other
    float theta_a = SafeACos(a.cosTheta); 
    float theta_b = SafeACos(b.cosTheta);
    float theta_d = AngleBetween(a.w, b.w);
    if (fminf(theta_d + theta_b, M_PIf) <= theta_a)
        return a;
    if (fminf(theta_d + theta_a, M_PIf) <= theta_b)
        return b;

    // Compute the spread angle of the merged cone, $\theta_o$
    float theta_o = (theta_a + theta_d + theta_b) / 2.0f;
    if (theta_o >= M_PIf)
        return DirectionCone::EntireSphere();

    // Find the merged cone's axis and return cone union
    float theta_r = theta_o - theta_a;
    Vector3f wr = cross(a.w, b.w);
    
    if (lengthSquared(wr) == 0)
        return DirectionCone::EntireSphere(); 
    //Vector3f w = Rotate(Degrees(theta_r), wr)(a.w);

    glm::mat4 rotate = glm::rotate(Degrees(theta_r), *(glm::vec3*)wr.data());
    glm::vec4 tmp = glm::vec4(a.w[0], a.w[1], a.w[2], 0.0f); 
    tmp = rotate * tmp;

    Vector3f w {tmp.x, tmp.y, tmp.z};
    return DirectionCone(w, cosf(theta_o));
}

inline void Inverse(LightBounds& lb) {
    lb.w *= -1.0f;
}

inline LightBounds Union(const LightBounds& a, const LightBounds& b) {
    if (a.phi <= 0) return b;
    if (b.phi <= 0) return a;

    DirectionCone cone = Union(DirectionCone(a.w, a.cosTheta_o), DirectionCone(b.w, b.cosTheta_o));
    float cosTheta_o = cone.cosTheta;
    float cosTheta_e = fminf(a.cosTheta_e, b.cosTheta_e);

    return LightBounds(Union(a.bounds, b.bounds), cone.w, a.phi + b.phi, cosTheta_o,
                    cosTheta_e, a.doubleSided | b.doubleSided);
}

#endif

} // namespace