#pragma once

#include <zeno/zty/mesh/Mesh.h>
#include <zeno/math/vec.h>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace zty {

struct MeshBevel {
    float fac = 0.1f;
    float smo = 0.25f;

    void operator()(Mesh &mesh) const;
};

}
ZENO_NAMESPACE_END
