#pragma once
#include "PathSamplingLaunch.h"

namespace zeno_path_sampling {
// EA FAST uniform scalar and 2D masks; source and license in sampling/fast.
// CMake embeds these assets so installed renderers need no external data path.
extern const unsigned char fastScalar[MaskSide*MaskSide*MaskDepth];
extern const unsigned char fastVector2[MaskSide*MaskSide*MaskDepth*2];
}
