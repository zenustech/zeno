#include <zeno/dop/dop.h>
#include <zeno/zty/mesh/Mesh.h>
#include <zeno/zty/mesh/MeshIO.h>
#include <fstream>


ZENO_NAMESPACE_BEGIN
namespace zty {
namespace {


static void ReadOBJMesh(dop::FuncContext *ctx) {
    auto path = value_cast<std::string>(ctx->inputs.at(0));
    auto mesh = std::make_shared<Mesh>();
    std::ifstream fin(path);
    [[unlikely]] if (!fin)
        throw ztd::format_error("OSError: cannot open file for read: {}", path);
    readMeshFromOBJ(fin, *mesh);
    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_DEFUN(ReadOBJMesh, {}, {{
    "mesh", "load mesh from .obj file",
}, {
    {"path", "string"},
}, {
    {"mesh", "Mesh"},
}});


}
}
ZENO_NAMESPACE_END
