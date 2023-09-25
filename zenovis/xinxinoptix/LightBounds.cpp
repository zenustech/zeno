#include "LightBounds.h"
#include "Sampling.h"
#include "vec_math.h"

namespace pbrt {
    
float LightBounds::importance(Vector3f p, Vector3f n) const {
    // Return importance for light bounds at reference point
    // Compute clamped squared distance to reference point
    Vector3f pc = (bounds.pMin + bounds.pMax) / 2; 
    float d2 = lengthSquared(p - pc);
    d2 = std::max(d2, length(bounds.diagonal()) / 2);

    // Define cosine and sine clamped subtraction lambdas
    auto cosSubClamped = [](float sinTheta_a, float cosTheta_a, float sinTheta_b, float cosTheta_b) -> float {
        if (cosTheta_a > cosTheta_b)
            return 1;
        return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
    };

    auto sinSubClamped = [](float sinTheta_a, float cosTheta_a, float sinTheta_b, float cosTheta_b) -> float {
        if (cosTheta_a > cosTheta_b)
            return 0;
        return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
    };

    // Compute sine and cosine of angle to vector _w_, $\theta_\roman{w}$
    Vector3f wi = normalize(p - pc);
    float cosTheta_w = dot(Vector3f(w), wi);
    if (doubleSided)
        cosTheta_w = std::abs(cosTheta_w);
    float sinTheta_w = SafeSqrt(1 - Sqr(cosTheta_w));

    // Compute $\cos\,\theta_\roman{\+b}$ for reference point
    float cosTheta_b = BoundSubtendedDirections(bounds, p).cosTheta;
    float sinTheta_b = SafeSqrt(1 - Sqr(cosTheta_b));

    // Compute $\cos\,\theta'$ and test against $\cos\,\theta_\roman{e}$
    float sinTheta_o = SafeSqrt(1 - Sqr(cosTheta_o));
    float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
    float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
    float cosThetap  = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
    if (cosThetap <= cosTheta_e)
        return 0;

    // Return final importance at reference point
    float importance = phi * cosThetap / d2;
    DCHECK(importance >= -1e-3);
    // Account for $\cos\theta_\roman{i}$ in importance at surfaces
    if (n[0]!=0 && n[1]!=0 && n[2]!=0) {
        float cosTheta_i = AbsDot(wi, n);
        float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
        float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
        importance *= cosThetap_i;
    }

    importance = fmaxf(importance, 0.0f);
    return importance;
}

}