#pragma once

#define OPENVDB_USE_VERSION_NAMESPACE
#define OPENVDB_VERSION_NAME vmocked

#include <memory>

#include <zinc/vec.h>

namespace openvdb {
    using Vec3s = zinc::vec3f;
    using Vec3I = zinc::vec3I;
    using Vec4s = zinc::vec4f;
    using Vec4I = zinc::vec4I;
    using Index = int;

    namespace math {
        struct Transform {
        };
    }
}
