#pragma once


#include <zeno/zty/mesh/Mesh.h>
#include <istream>
#include <ostream>


ZENO_NAMESPACE_BEGIN
namespace zty {


void readMeshFromOBJ(std::istream &in, Mesh &mesh);
void writeMeshToOBJ(std::ostream &in, Mesh const &mesh);


}
ZENO_NAMESPACE_END
