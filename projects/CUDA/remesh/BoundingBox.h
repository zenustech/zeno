// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <zeno/utils/vec.h>

namespace zeno {
namespace pmp {

class BoundingBox {
public:
    BoundingBox()
        : min_(std::numeric_limits<float>::max()),
          max_(-std::numeric_limits<float>::max())
    {}

    BoundingBox(const std::vector<vec3f>& points) {
        min_ = vec3f(std::numeric_limits<float>::max());
        max_ = vec3f(-std::numeric_limits<float>::max());
        for (auto &it: points) {
            for (int i = 0; i < 3; ++i) {
                if (it[i] < min_[i])
                    min_[i] = it[i];
                if (it[i] > max_[i])
                    max_[i] = it[i];
            }
        }
    }

    BoundingBox& operator+=(const vec3f& p) {
        for (int i = 0; i < 3; ++i) {
            if (p[i] < min_[i])
                min_[i] = p[i];
            if (p[i] > max_[i])
                max_[i] = p[i];
        }
        return *this;
    }

    vec3f& min() { return min_; }
    vec3f& max() { return max_; }
    vec3f center() const { return 0.5f * (min_ + max_); }
    float size() const {
        if (max_[0] < min_[0] || max_[1] < min_[1] || max_[2] < min_[2]) {
            return 0.0;
        }
        return distance(max_, min_); 
    }
    bool collide(const BoundingBox& b) const {
        return b.min_[0]<max_[0] && b.max_[0]>min_[0] &&
            b.min_[1]<max_[1] && b.max_[1]>min_[1] &&
            b.min_[2]<max_[2] && b.max_[2]>min_[2];
    }

private:
    vec3f min_, max_;
};

} // namespace pmp
} // namespace zeno
