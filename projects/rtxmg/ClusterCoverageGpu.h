#pragma once
#include "ClusterCoverage.h"
#include <memory>

namespace zeno::cluster::coverage {
// Offline accelerator only. It does not own renderer streams or modify caches.
class GpuBatch {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    explicit GpuBatch(bool convex=false);
    ~GpuBatch();
    void add(uint32_t group, const Surface& coarse, const std::vector<Triangle>& fine);
    size_t triangles()const;
    void flush(std::vector<double>& localBounds);
};
}
