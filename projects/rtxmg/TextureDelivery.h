#pragma once

namespace zeno_cluster_device {
// Bounded, sampled demand importance. page stores virtual page ID + 1;
// zero is empty. Counts change admission order, never the desired footprint.
inline constexpr unsigned int TextureDemandSlots=16384;
struct TextureDemand {
    unsigned int page, fine, parent;
};
}
