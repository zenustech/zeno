#include <zeno/zty/mesh/MeshCutter.h>
#include <zeno/zty/mesh/MeshTriangulate.h>
#include <zeno/zty/mesh/MeshIO.h>
#include <fstream>

int main()
{
    zty::Mesh mesh1;
    {
        std::ifstream ifs("models/Pig_Head.obj");
        if (!ifs) throw std::runtime_error("src mesh file read fail");
        readMeshFromOBJ(ifs, mesh1);
    }

    zty::Mesh mesh2;
    {
        std::ifstream ifs("models/cube.obj");
        if (!ifs) throw std::runtime_error("cut mesh file read fail");
        readMeshFromOBJ(ifs, mesh2);
    }

    zty::Mesh mesh3;
    {
        zty::MeshCutter mcut;
        mcut.dispatch(mesh1, mesh2);
        mcut.selectComponents(zty::MeshCutter::CompType::fragment);
        mcut.getComponent(0, mesh3);
    }


    {
        std::ofstream ofs("/tmp/a.obj");
        if (!ofs) throw std::runtime_error("cut mesh file write fail");
        writeMeshToOBJ(ofs, mesh3);
    }

    return 0;
}
