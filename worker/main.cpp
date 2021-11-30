#include <zeno/zty/mesh/MeshCutter.h>
#include <zeno/zty/mesh/MeshTriangulate.h>
#include <zeno/zty/mesh/MeshIO.h>
#include <fstream>

int main()
{
    zty::Mesh mesh1;
    {
        std::ifstream ifs("models/monkey.obj");
        if (!ifs) throw std::runtime_error("src mesh file");
        readMeshFromOBJ(ifs, mesh1);
    }

    zty::Mesh mesh2;
    {
        std::ifstream ifs("models/plane.obj");
        if (!ifs) throw std::runtime_error("cut mesh file");
        readMeshFromOBJ(ifs, mesh2);
    }

    zty::Mesh mesh3;
    {
        zty::MeshCutter mcut(mesh1, mesh2);
        mcut.selectComponents(zty::MeshCutter::CompType::all);
        mcut.getComponent(0, mesh3);
    }


    {
        std::ofstream ofs("/tmp/a.obj");
        if (!ofs) throw std::runtime_error("cut mesh file");
        writeMeshToOBJ(ofs, mesh3);
    }

    return 0;
}
