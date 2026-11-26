#pragma once

// POD shared with the embedded NVRTC sources. Zero selects the original path.
namespace zeno_path_sampling {
// Preserve the FAST identifier used by existing renderer recordings.
enum Mode : unsigned { Legacy=0, FAST=3 };
constexpr unsigned MaskSide=128;
constexpr unsigned MaskDepth=32;
struct Launch {
    const unsigned char* scalar;
    const unsigned char* vector2;
    unsigned mode;
    unsigned sampleIndex;
    unsigned seed;
    unsigned independentTiles;
};
}
