#pragma once

#include <zeno/utils/vec.h>
#include <variant>
#include <vector>

namespace zeno {

using AttributeArray = std::variant<std::vector<vec3f>, std::vector<float>>;

}
