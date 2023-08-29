#pragma once


#include <optix.h>
#include <sutil/vec_math.h>

#include "Sampling.h"
#include "LightBounds.h"
#include "optixPathTracer.h"

#ifndef __CUDACC_RTC__ 

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <tuple>

#include <map>
#include <tuple>
#include <optional>
#include <sutil/Exception.h>

#endif

namespace pbrt {

// OctahedralVector Definition
struct OctahedralVector {
  public:
    // OctahedralVector Public Methods
    OctahedralVector() = default;

    OctahedralVector(Vector3f vvv) {

        auto& v = reinterpret_cast<float3&>(vvv);

        v /= fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
        if (v.z >= 0) {
            x = Encode(v.x);
            y = Encode(v.y);
        } else {
            // Encode octahedral vector with $z < 0$
            x = Encode((1 - fabsf(v.y)) * Sign(v.x));
            y = Encode((1 - fabsf(v.x)) * Sign(v.y));
        }
    }

    explicit operator Vector3f() const {
        float3 v;
        v.x = -1 + 2 * (x / 65535.f);
        v.y = -1 + 2 * (y / 65535.f);
        v.z = 1 - (fabsf(v.x) + fabsf(v.y));
        // Reparameterize directions in the $z<0$ portion of the octahedron
        if (v.z < 0) {
            float xo = v.x;
            v.x = (1 - fabsf(v.y)) * Sign(xo);
            v.y = (1 - fabsf(xo)) * Sign(v.y);
        }

        return normalize(reinterpret_cast<Vector3f&>(v));
    }

  private:
    // OctahedralVector Private Methods
    static float Sign(float v) { return copysignf(1.f, v); }

    static uint16_t Encode(float f) {
        return roundf(clamp((f + 1) / 2.0f, 0.0f, 1.0f) * 65535.f);
    }

    // OctahedralVector Private Members
    uint16_t x, y;
};

// CompactLightBounds Definition
struct CompactLightBounds {
  public:
    // CompactLightBounds Public Methods
    CompactLightBounds() = default;

    CompactLightBounds(const LightBounds &lb, const Bounds3f &allb) : w(normalize(lb.w)), phi(lb.phi)
    {
        this->meta = {
            QuantizeCos(lb.cosTheta_o),
            QuantizeCos(lb.cosTheta_e),
            lb.doubleSided
        };
        // Quantize bounding box into _qb_
        for (int c = 0; c < 3; ++c) {
            qb[0][c] =
                floorf(QuantizeBounds(lb.bounds[0][c], allb.pMin[c], allb.pMax[c]));
            qb[1][c] =
                ceilf(QuantizeBounds(lb.bounds[1][c], allb.pMin[c], allb.pMax[c]));
        }
    }

    bool isDoubleSided() const { return meta.doubleSided; }
    
    float CosTheta_o() const { return 2 * (meta.qCosTheta_o / 32767.f) - 1; }
    float CosTheta_e() const { return 2 * (meta.qCosTheta_e / 32767.f) - 1; }

    inline float lerp(float t, const float a, const float b) const {
        return a + t*(b-a);
    }

    Bounds3f Bounds(const Bounds3f &allb) const {

        const auto& pMin = reinterpret_cast<const float3&>(allb.pMin);
        const auto& pMax = reinterpret_cast<const float3&>(allb.pMax);

        return {Vector3f(lerp(qb[0][0] / 65535.f, pMin.x, pMax.x),
                            lerp(qb[0][1] / 65535.f, pMin.y, pMax.y),
                            lerp(qb[0][2] / 65535.f, pMin.z, pMax.z)),
                Vector3f(lerp(qb[1][0] / 65535.f, pMin.x, pMax.x),
                            lerp(qb[1][1] / 65535.f, pMin.y, pMax.y),
                            lerp(qb[1][2] / 65535.f, pMin.z, pMax.z))};
    }

    float Importance(Vector3f p, Vector3f n, const Bounds3f &allb) const {
        Bounds3f bounds = Bounds(allb);
        float cosTheta_o = CosTheta_o(), cosTheta_e = CosTheta_e();
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        Vector3f pc = (bounds.pMin + bounds.pMax) / 2;
        float d2 = lengthSquared(p - pc);
        d2 = fmaxf(d2, length(bounds.diagonal()) / 2);

        // Define cosine and sine clamped subtraction lambdas
        auto cosSubClamped = [](float sinTheta_a, float cosTheta_a, float sinTheta_b, float cosTheta_b) -> float {
            if (cosTheta_a > cosTheta_b)
                return 1;
            return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
        }; // cos( theta_a - theta_b )

        auto sinSubClamped = [](float sinTheta_a, float cosTheta_a, float sinTheta_b, float cosTheta_b) -> float {
            if (cosTheta_a > cosTheta_b)
                return 0;
            return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
        }; // sin( theta_a - theta_b )

        // Compute sine and cosine of angle to vector _w_, $\theta_\roman{w}$
        Vector3f wi = normalize(p - pc);
        float cosTheta_w = dot(Vector3f(w), wi);
        if (meta.doubleSided)
            cosTheta_w = fabsf(cosTheta_w);
        float sinTheta_w = SafeSqrt(1 - Sqr(cosTheta_w));

        // Compute $\cos\,\theta_\roman{\+b}$ for reference point
        float cosTheta_b = BoundSubtendedDirections(bounds, p).cosTheta;
        float sinTheta_b = SafeSqrt(1 - Sqr(cosTheta_b));

        // Compute $\cos\,\theta'$ and test against $\cos\,\theta_\roman{e}$
        float sinTheta_o = SafeSqrt(1 - Sqr(cosTheta_o));
        float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        float cosThetap  = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
        if (cosThetap <= cosTheta_e || cosThetap < 0.0f)
            return 0;

        // Return final importance at reference point
        float importance = phi * cosThetap / d2;
        DCHECK(importance >= -1e-3f);

        if (n[0]!=0 && n[1]!=0 && n[2]!=0) {

            float cosTheta_i = AbsDot(wi, n);
            float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
            float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
            importance *= cosThetap_i;
        }

        importance = fmaxf(importance, 0);
        return importance;
    }

  private:
    // CompactLightBounds Private Methods
    static unsigned int QuantizeCos(float c) {
        DCHECK(c >= -1 && c <= 1);
        return floorf(32767.f * ((c + 1) / 2));
    }

    static float QuantizeBounds(float c, float min, float max) {
        DCHECK(c >= min && c <= max);
        if (min == max)
            return 0;
        return 65535.f * clamp((c - min) / (max - min), 0.0f, 1.0f);
    }

    // CompactLightBounds Private Members
    OctahedralVector w;
    float phi = 0;
    struct {
        unsigned int qCosTheta_o:15;
        unsigned int qCosTheta_e:15;
        bool doubleSided:1;
    } meta;
    uint16_t qb[2][3];
};

// LightBVHNode Definition
struct alignas(32) LightTreeNode {

    LightTreeNode() = default;

    static LightTreeNode MakeLeaf(unsigned int lightIndex, const CompactLightBounds &cb) {
        return LightTreeNode{cb, {lightIndex, 1}};
    }

    static LightTreeNode MakeInterior(unsigned int child1Index,
                                     const CompactLightBounds &cb) {
        return LightTreeNode{cb, {child1Index, 0}};
    }

    CompactLightBounds lightBounds;
    struct {
        unsigned int childOrLightIndex : 31;
        unsigned int isLeaf : 1;
    } meta;
};

struct SelectedLight {
    uint32_t lightIdx;
    float prob = 0.0f;
};

// BVHLightSampler Definition
struct LightTreeSampler {
  public:

#ifndef __CUDACC_RTC__ 

    LightTreeSampler(std::vector<GenericLight> &lights);

    inline void upload(CUdeviceptr &lightBitTrailsPtr, CUdeviceptr &lightTreeNodesPtr) 
    {
        {
            size_t byte_length = sizeof( lightBitTrails[0] ) * lightBitTrails.size();

            CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &lightBitTrailsPtr ), byte_length) );
            CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lightBitTrailsPtr ),
                                   lightBitTrails.data(), byte_length, cudaMemcpyHostToDevice) );
        }
        {
            size_t byte_length = sizeof( nodes[0] ) * nodes.size();

            CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &lightTreeNodesPtr ), byte_length) );
            CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lightTreeNodesPtr ),
                                   nodes.data(), byte_length, cudaMemcpyHostToDevice) );
        }
    }

#endif

    inline Bounds3f bounds() { return rootBounds; }

    inline SelectedLight sample(float u, const Vector3f& p, const Vector3f& n) 
    {
        // Traverse light BVH to sample light
        #ifndef __CUDACC_RTC__
            if (nodes.empty()) return SelectedLight();
        #else 
            if (nullptr == nodes) return SelectedLight();
        #endif

        int nodeIndex = 0;
        float pmf = 1.0f;

        while (true) {
            // Process light BVH node for light sampling
            LightTreeNode& node = nodes[nodeIndex];
            if (!node.meta.isLeaf) {
                // Compute light BVH child node importances
                const LightTreeNode *child0 = &nodes[nodeIndex + 1];
                const LightTreeNode *child1 = &nodes[node.meta.childOrLightIndex];
                
                float ci[3] = { 0.0f,
                    child0->lightBounds.Importance(p, n, rootBounds),
                    child1->lightBounds.Importance(p, n, rootBounds) };

                DCHECK(ci[1] >= 0 && ci[2] >= 0);
                
                if (ci[1] == 0 && ci[2] == 0)
                    return {};

                // Randomly sample light BVH child node
                float nodePMF;

                int child = [&u, &ci, &nodePMF]() {
                    
                    auto sum = ci[1] + ci[2];
                    auto up = u * sum;

                    int pick_idx = up<ci[1] ? 1:2;

                    nodePMF = ci[pick_idx] / sum;
                    u = (up - ci[pick_idx-1]) / ci[pick_idx];

                    return pick_idx - 1;
                }();

                pmf *= nodePMF;
                nodeIndex = (child == 0) ? (nodeIndex + 1) : node.meta.childOrLightIndex;

            } else {
                // Confirm light has nonzero importance before returning light sample
                if (nodeIndex >= 0 && node.lightBounds.Importance(p, n, rootBounds) > 0) {
                    return SelectedLight{ node.meta.childOrLightIndex, pmf};
                }
                
                return {};
            }
        }

        return {};
    }

    float PMF(const Vector3f &p, const Vector3f &n, const uint32_t lightIdx) const {
        
        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = lightBitTrails[lightIdx];

        int nodeIndex = 0;
        float pmf = 1.0f;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightTreeNode *node = &nodes[nodeIndex];
            if (node->meta.isLeaf) {
                DCHECK(lightIdx == node->meta.childOrLightIndex);
                return pmf;
            }

            // Compute child importances and update PMF for current node
            const LightTreeNode *child0 = &nodes[nodeIndex + 1];
            const LightTreeNode *child1 = &nodes[node->meta.childOrLightIndex];
            float ci[2] = {child0->lightBounds.Importance(p, n, rootBounds),
                           child1->lightBounds.Importance(p, n, rootBounds)};
            DCHECK(ci[bitTrail & 1] > 0);
            pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->meta.childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

  private:

#ifndef __CUDACC_RTC__

    std::pair<int, LightBounds> buildTree(std::vector<std::pair<int, LightBounds>> &bvhLights,
                                          int start, int end, uint32_t bitTrail, int depth);
#endif

    float EvaluateCost(const LightBounds &b, const Bounds3f &bounds, int dim) const {
        // Evaluate direction bounds measure for _LightBounds_
        float theta_o = acosf(b.cosTheta_o);
        float theta_e = acosf(b.cosTheta_e);
        float theta_w = fminf(theta_o + theta_e, M_PIf);
        float sinTheta_o = SafeSqrt(1 - Sqr(b.cosTheta_o));
        float M_omega = 2 * M_PIf * (1 - b.cosTheta_o) +
                        M_PIf / 2 *
                            (2 * theta_w * sinTheta_o - cosf(theta_o - 2 * theta_w) -
                             2 * theta_o * sinTheta_o + b.cosTheta_o);

        float Kr = MaxComponentValue(bounds.diagonal()) / bounds.diagonal()[dim];

        return b.phi * M_omega * Kr * b.bounds.area();
    }

// BVHLightSampler Private Members
#ifndef __CUDACC_RTC__    
    std::vector<uint32_t> lightBitTrails{};
    std::vector<LightTreeNode>     nodes{};
#else 
    uint32_t *lightBitTrails;
    LightTreeNode     *nodes; 
#endif
    Bounds3f rootBounds;
};

} // namespace