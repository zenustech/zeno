#pragma once


#include <zeno/zty/mesh/Mesh.h>
#include <memory>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct MeshCutter {
    struct Impl;
    std::unique_ptr<Impl> impl;

    MeshCutter(Mesh const &mesh1, Mesh const &mesh2);
    Mesh getComponent(size_t i);
    ~MeshCutter();
};


}
ZENO_NAMESPACE_END
