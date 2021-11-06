#pragma once

#include <istream>
#include <ostream>

ZENO_NAMESPACE_BEGIN
namespace types {

void readMeshFromOBJ(std::istream &in, Mesh &mesh);
void writeMeshToOBJ(std::ostream &in, Mesh const &mesh);

}
ZENO_NAMESPACE_END
