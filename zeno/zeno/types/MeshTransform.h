#pragma once


#include <zeno/types/Mesh.h>


ZENO_NAMESPACE_BEGIN
namespace types {


void transformMesh
    ( Mesh &mesh
    , math::vec3f const &translate
    , math::vec3f const &scaling
    , math::vec4f const &rotation
    );


}
ZENO_NAMESPACE_END
