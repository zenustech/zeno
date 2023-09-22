#include "LightTree.h"
#include "Sampling.h"
#include "optixPathTracer.h"

#include <algorithm>

namespace pbrt {
// BVHLightSampler Method Definitions
LightTreeSampler::LightTreeSampler(std::vector<GenericLight> &lights) {

    lightBitTrails.resize(lights.size(), 0u);
    nodes.reserve(lights.size());

    std::vector<std::pair<int, LightBounds>> bvhLights{};
    bvhLights.reserve(lights.size() * 2);

    for (size_t i = 0; i < lights.size(); ++i) {
        
        auto& light = lights[i];
        LightBounds lightBounds = light.bounds();

        if (lightBounds.phi > 0) {
            bvhLights.push_back(std::make_pair(i, lightBounds));
            rootBounds = Union(rootBounds, lightBounds.bounds);
        }
    }
    if (!bvhLights.empty())
        buildTree(bvhLights, 0, bvhLights.size(), 0, 0);
}

std::pair<int, LightBounds> LightTreeSampler::buildTree(
            std::vector<std::pair<int, LightBounds>> &bvhLights, 
            int start, int end, uint32_t bitTrail, int depth) {
        
    DCHECK(start < end);
    // Initialize leaf node if only a single light remains
    if (end - start == 1) {
        int nodeIndex = nodes.size();
        CompactLightBounds cb(bvhLights[start].second, rootBounds);
        int lightIndex = bvhLights[start].first;
        nodes.push_back(LightTreeNode::MakeLeaf(lightIndex, cb));

        lightBitTrails.at(lightIndex) = bitTrail;
        return {nodeIndex, bvhLights[start].second};
    }

    // Choose split dimension and position using modified SAH
    // Compute bounds and centroid bounds for lights
    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = bvhLights[i].second;
        bounds = Union(bounds, lb.bounds);
        centroidBounds = Union(centroidBounds, lb.centroid());
    }

    float minCost = INFINITY;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;
    for (int dim = 0; dim < 3; ++dim) {
        // Compute minimum cost bucket for splitting along dimension _dim_
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
            continue;
        // Compute _LightBounds_ for each bucket
        LightBounds bucketLightBounds[nBuckets];
        for (int i = start; i < end; ++i) {
            Vector3f pc = bvhLights[i].second.centroid();
            int b = nBuckets * centroidBounds.offset(pc)[dim];
            if (b == nBuckets)
                b = nBuckets - 1;
            DCHECK(b >= 0);
            DCHECK(b < nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], bvhLights[i].second);
        }

        // Compute costs for splitting lights after each bucket
        float cost[nBuckets - 1]{};
        for (int i = 0; i < (nBuckets - 1); ++i) {
            // Find _LightBounds_ for lights below and above bucket split
            LightBounds b0, b1;
            for (int j = 0; j <= i; ++j)
                b0 = Union(b0, bucketLightBounds[j]);
            for (int j = (i + 1); j < nBuckets; ++j)
                b1 = Union(b1, bucketLightBounds[j]);

            // Compute final light split cost for bucket
            cost[i] = EvaluateCost(b0, bounds, dim) + EvaluateCost(b1, bounds, dim);
        }

        // Find light split that minimizes SAH metric
        for (int i = 1; i < (nBuckets - 1); ++i) {
            if (cost[i] > 0 && cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1)
        mid = (start + end) / 2;
    else {
        const auto *pmid = std::partition(
            &bvhLights[start], &bvhLights[end - 1] + 1,
            [=](const std::pair<int, LightBounds> &l) {
                int b = nBuckets *
                        centroidBounds.offset(l.second.centroid())[minCostSplitDim];
                if (b == nBuckets)
                    b = nBuckets - 1;
                DCHECK(b >= 0);
                DCHECK(b < nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &bvhLights[0];
        if (mid == start || mid == end)
            mid = (start + end) / 2;
        DCHECK(mid > start && mid < end);
    }

    // Allocate interior _LightBVHNode_ and recursively initialize children
    int nodeIndex = nodes.size();
    nodes.push_back(LightTreeNode());
    DCHECK(depth < 64);
    std::pair<int, LightBounds> child0 =
        buildTree(bvhLights, start, mid, bitTrail, depth + 1);
    DCHECK(nodeIndex + 1 == child0.first);
    std::pair<int, LightBounds> child1 =
        buildTree(bvhLights, mid, end, bitTrail | (1u << depth), depth + 1);

    // Initialize interior node and return node index and bounds
    LightBounds lb = Union(child0.second, child1.second);
    CompactLightBounds cb(lb, rootBounds);
    nodes[nodeIndex] = LightTreeNode::MakeInterior(child1.first, cb);
    return {nodeIndex, lb};
}

}