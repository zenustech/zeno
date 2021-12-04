#pragma once


#include <zeno/zty/mesh/Mesh.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void meshSubdivisionSimple(Mesh &mesh);
void meshSubdivisionCatmull(Mesh &mesh, int numIters = 1);


}
ZENO_NAMESPACE_END
