#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/MeshIO.h>
#include <fstream>


ZENO_NAMESPACE_BEGIN
namespace types {
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
