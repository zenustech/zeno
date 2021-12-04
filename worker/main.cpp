#include <zeno/zty/mesh/MeshTriangulate.h>
#include <zeno/zty/mesh/MeshSubdivision.h>
#include <zeno/zty/mesh/MeshIO.h>
#include <zeno/zty/mesh/DCEL.h>
#include <fstream>

int main()
{
    zty::Mesh mesh1;
    {
        std::ifstream ifs("models/cube.obj");
        if (!ifs) throw std::runtime_error("mesh1 file read fail");
        readMeshFromOBJ(ifs, mesh1);
        meshSubdivisionSimple(mesh1);
    }

    {
        zty::DCEL dcel1(mesh1);
        mesh1 = (zty::Mesh)dcel1;
    }


    {
        std::ofstream ofs("/tmp/a.obj");
        if (!ofs) throw std::runtime_error("mesh1 file write fail");
        writeMeshToOBJ(ofs, mesh1);
    }

    return 0;
}
