#include <zeno/zty/mesh/MeshSubdivision.h>
#include <zeno/zty/mesh/DCEL.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void meshSubdivisionSimple(Mesh &mesh)
{
    // TODO
}


void meshSubdivisionCatmull(Mesh &mesh, int numIters)
{
    DCEL dcel(mesh);
    for (int i = 0; i < numIters; i++) {
        dcel = dcel.subdivision();
    }
    mesh = (Mesh)dcel;
}


}
ZENO_NAMESPACE_END
