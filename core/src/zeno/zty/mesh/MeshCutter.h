#pragma once


#include <zeno/zty/mesh/Mesh.h>
#include <memory>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct MeshCutter {
    struct Impl;
    std::unique_ptr<Impl> impl;

    enum class CompType {
        all,
        fragment,
        patch,
        seam,
        input,
    };

    explicit MeshCutter(bool debugMode = true);
    void dispatch(Mesh const &mesh1, Mesh const &mesh2);
    void selectComponents(CompType compType) const;
    size_t getNumComponents() const;
    void getComponent(size_t i, Mesh &mesh) const;
    ~MeshCutter();
};


}
ZENO_NAMESPACE_END
