#include <zeno/zty/mesh/MeshBevel.h>
#include <zeno/zty/mesh/MeshTriangulate.h>
#include <zeno/zty/mesh/MeshIO.h>
#include <fstream>

int main()
{
    zty::Mesh mesh1;
    {
        std::ifstream ifs("models/cube.obj");
        if (!ifs) throw std::runtime_error("mesh1 file read fail");
        readMeshFromOBJ(ifs, mesh1);
    }

    {
        zty::MeshBevel bevel;
        bevel(mesh1);
    }


    {
        std::ofstream ofs("/tmp/a.obj");
        if (!ofs) throw std::runtime_error("mesh1 file write fail");
        writeMeshToOBJ(ofs, mesh1);
    }

    return 0;
}
