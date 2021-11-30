#pragma once


#include <zeno/zty/mesh/Mesh.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void transformMesh
    ( Mesh &mesh
    , math::vec3f const &translate
    , math::vec3f const &scaling
    , math::vec4f const &rotation
    );


}
ZENO_NAMESPACE_END
