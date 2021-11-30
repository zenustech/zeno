#include <zeno/zty/mesh/MeshCutter.h>
#include <zeno/zty/mesh/MeshIO.h>
#include <fstream>

int main()
{
    zty::Mesh mesh1;
    {
        std::ifstream ifs("models/monkey.obj");
        if (!ifs) throw std::runtime_error("src mesh file");
        zty::readMeshFromOBJ(ifs, mesh1);
    }

    zty::Mesh mesh2;
    {
        std::ifstream ifs("models/plane.obj");
        if (!ifs) throw std::runtime_error("cut mesh file");
        zty::readMeshFromOBJ(ifs, mesh2);
    }

    zty::Mesh mesh3;
    {
        zty::MeshCutter mcut(mesh1, mesh2);
        mcut.selectComponents(zty::MeshCutter::CompType::all);
        mesh3 = mcut.getComponent(0);
    }

    {
        std::ofstream ofs("/tmp/a.obj");
        zty::writeMeshToOBJ(ofs, mesh3);
    }

    return 0;
}
