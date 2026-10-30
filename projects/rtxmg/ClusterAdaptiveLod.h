#pragma once
#include <algorithm>
#include <cmath>

namespace zeno::cluster {
// Streaming policy only. The renderer publishes complete cuts independently.
// Rates are expressed relative to 60 Hz, but advance with elapsed time.
class AdaptiveLod {
    float value_{}, load_{}, recoveryFloor_{};
    bool observed_{}, recovering_{};
  public:
    void reset(float requested) {
        value_ = recoveryFloor_ = requested;
        load_ = 0.f;
        observed_ = recovering_ = false;
    }
    void viewChanged(float requested) {
        recoveryFloor_ = requested;
        recovering_ = false;
    }
    float value() const { return value_; }
    float load() const { return load_; }
    float recoveryFloor() const { return recoveryFloor_; }
    bool pending(float requested, float occupancy, bool adaptive, bool canCoarsen, bool blocked) const {
        if (!adaptive) return value_ != requested;
        if (value_ < requested) return true;
        return (canCoarsen && (blocked || occupancy > .85f || load_ > .85f)) ||
               (value_ > std::max(requested, recoveryFloor_) && (occupancy < .70f || load_ < .70f));
    }
    float advance(float requested, float occupancy, double seconds, bool adaptive,
                  bool canCoarsen, bool blocked) {
        const float frames = float(std::clamp(seconds, 0., .25) * 60.);
        if (!observed_) { load_ = occupancy; observed_ = true; }
        else load_ += (occupancy-load_) * (1.f-std::pow(.95f, frames));
        if (!adaptive) {
            value_ = recoveryFloor_ = requested;
            recovering_ = false;
            return value_;
        }
        value_ = std::max(value_, requested);
        if (canCoarsen && (blocked || load_ > .85f)) {
            value_ = std::min(value_ * std::pow(1.02f, frames), std::max(requested,1.e20f));
            // A discrete LOD step can jump straight from <70% to >85%.
            // Remember that failed recovery for this view, rather than
            // repeatedly loading and evicting the same cut at a fixed camera.
            if (recovering_) recoveryFloor_ = std::max(recoveryFloor_, value_);
        } else if (load_ < .70f && occupancy < .85f) {
            const float floor = std::max(requested, recoveryFloor_);
            if (value_ > floor) {
                value_ = std::max(floor, value_ * std::pow(.995f, frames));
                recovering_ = true;
            }
        }
        return value_;
    }
};
} // namespace zeno::cluster
