#pragma once

// Built on the host, uploaded in Params, sampled/evaluated on the device.
// Sampling and reverse-PDF evaluation use the same unconditional group PMF.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define LIGHT_SELECTION_HD __host__ __device__
#else
#define LIGHT_SELECTION_HD
#endif

namespace LightSelection {

enum class Group { Distant, Portal, Environment, Local, None };

LIGHT_SELECTION_HD inline bool positiveFinite(float value)
{
    return value > 0.0f && value <= 3.402823466e+38f;
}

LIGHT_SELECTION_HD inline float unitSample(float u)
{
    // RNG endpoints and roundoff must not select a zero-width interval or count.
    return (u <= 0.0f) ? 0.0f : (u < 1.0f ? u : 0.9999999403953552f);
}

struct Sample {
    Group group = Group::None;
    float pmf = 0.0f;
    float u = 0.0f; // Uniform sample remapped into the selected group's interval.
};

class Distribution {
    // No default member initializer: Params lives in OptiX __constant__ memory
    // and must not require dynamic initialization. The host fills every entry.
    float probabilities[4];

public:
    Distribution() = default;

#ifndef __CUDACC_RTC__
    // Construction is host-only. Scale before summation to avoid overflow for
    // large finite HDR strengths; invalid/nonpositive weights disable a group.
    Distribution(float localWeight, float distantWeight, float portalWeight, float environmentWeight)
    {
        const float weights[] = {distantWeight, portalWeight, environmentWeight, localWeight};
        float maximum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            probabilities[i] = positiveFinite(weights[i]) ? weights[i] : 0.0f;
            if (probabilities[i] > maximum) maximum = probabilities[i];
        }
        if (maximum == 0.0f) return;
        float total = 0.0f;
        for (int i = 0; i < 4; ++i) {
            probabilities[i] /= maximum;
            total += probabilities[i];
        }
        for (int i = 0; i < 4; ++i) probabilities[i] /= total;
    }
#endif

    LIGHT_SELECTION_HD float pmf(Group group) const
    {
        return group == Group::None ? 0.0f : probabilities[int(group)];
    }

    LIGHT_SELECTION_HD Sample sample(float u) const
    {
        u = unitSample(u);
        float start = 0.0f;
        Sample last;
        for (int i = 0; i < 4; ++i) {
            const float p = probabilities[i];
            if (p <= 0.0f) continue;
            last = {Group(i), p, unitSample((u - start) / p)};
            if (u < start + p) return last;
            start += p;
        }
        // Protect the final interval against floating-point summation roundoff.
        return last;
    }
};

// Keep resource eligibility next to the host policy, testable without GPU data.
// Portal maps are baked from the HDR texture without sky_strength. They share
// the HDR family's selection budget; multiplying their emitted radiance by the
// current sky_strength is a separate operation in DirectLighting.
#ifndef __CUDACC_RTC__
template <typename LaunchParams>
inline Distribution fromScene(const LaunchParams& p, bool distant, bool portal)
{
    const bool local = p.num_lights > 0 && p.lights != nullptr && p.lightTreeSampler != 0;
    const bool hdr = p.usingHdrSky && p.sky_texture != 0 && p.skynx > 0 && p.skyny > 0 &&
        positiveFinite(p.sky_strength) && positiveFinite(p.envavg);
    const bool environment = hdr && p.skycdf != nullptr && p.sky_start != nullptr;
    portal = portal && hdr;
    const int hdrTechniques = int(environment) + int(portal);
    const float hdrWeight = hdrTechniques ? p.sky_strength / hdrTechniques : 0.0f;
    return Distribution(local ? 1.0f : 0.0f, distant ? 1.0f : 0.0f,
                        portal ? hdrWeight : 0.0f, environment ? hdrWeight : 0.0f);
}
#endif

struct DiscreteSample {
    int index = -1;
    float pmf = 0.0f;
};

// CDFs store inclusive cumulative probabilities, with exactly count entries.
// Upper-bound search skips zero-power entries, including when u is exactly zero.
LIGHT_SELECTION_HD inline DiscreteSample sampleCDF(const float* cdf, int count, float u)
{
    if (cdf == nullptr || count <= 0 || !positiveFinite(cdf[count - 1])) return {};
    const float total = cdf[count - 1];
    const float target = unitSample(u) * total;
    int low = 0, high = count - 1;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (cdf[mid] <= target) low = mid + 1;
        else high = mid;
    }
    const float mass = cdf[low] - (low > 0 ? cdf[low - 1] : 0.0f);
    return positiveFinite(mass) ? DiscreteSample{low, mass / total} : DiscreteSample{};
}

} // namespace LightSelection

#undef LIGHT_SELECTION_HD
