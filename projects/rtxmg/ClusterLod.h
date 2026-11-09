#pragma once
#include <cmath>

namespace zeno::cluster {
// A zoom lens should spend its extra detail on the image, rather than scaling
// the entire world's demand until the residency budget cancels the zoom.
// Side planes are relative to the eye, normalized, and include a guard band.
// A zero peripheralProjection preserves the omnidirectional diagnostic path.
struct ZoomView {
    float planes[4][3]{};
    float feather{}, peripheralProjection{};
    bool operator==(const ZoomView &other) const {
        if (feather != other.feather || peripheralProjection != other.peripheralProjection)
            return false;
        for (int p = 0; p < 4; ++p)
            for (int a = 0; a < 3; ++a)
                if (planes[p][a] != other.planes[p][a]) return false;
        return true;
    }
    bool operator!=(const ZoomView &other) const { return !(*this == other); }
};
inline ZoomView makeZoomView(const float *right, const float *up, const float *forward,
                             float tanHalfX, float tanHalfY, float peripheralProjection,
                             float horizontalShift = 0.f, float verticalShift = 0.f) {
    ZoomView view;
    view.peripheralProjection = peripheralProjection;
    view.feather = .5f * fminf(tanHalfX, tanHalfY);
    for (int p = 0; p < 4; ++p) {
        const float *axis = p < 2 ? right : up;
        const float shift = p < 2 ? horizontalShift : verticalShift;
        const float tangent = (1.2f + (p % 2 ? 2.f : -2.f) * shift) * (p < 2 ? tanHalfX : tanHalfY);
        const float invLength = 1.f / sqrtf(1.f + tangent * tangent);
        for (int a = 0; a < 3; ++a)
            view.planes[p][a] = (forward[a] * tangent + (p % 2 ? -axis[a] : axis[a])) * invLength;
    }
    return view;
}
#if defined(__CUDACC__)
__host__ __device__
#endif
inline float projectionForSphere(const ZoomView &view, float x, float y, float z, float radius, float distance, float projection) 
{
    if (!(view.peripheralProjection > 0.f) || projection <= view.peripheralProjection)
        return projection;
    float outside = 0.f;
    for (int p = 0; p < 4; ++p)
        outside = fmaxf(outside, -(view.planes[p][0] * x + view.planes[p][1] * y +
                                   view.planes[p][2] * z) - radius);
    float blend = fminf(outside / fmaxf(distance * view.feather, .01f), 1.f);
    blend = blend * blend * (3.f - 2.f * blend);
    // Keep a complete, ordinary wide-view cut outside the zoom region. Never
    // cull: shadow and reflection rays must still intersect those surfaces.
    return projection + (view.peripheralProjection - projection) * blend;
}
} // namespace zeno::cluster
