#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/OBJMeshIO.h>
#include <fstream>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void WriteOBJMesh(dop::FuncContext *ctx) {
    auto mesh = pointer_cast<Mesh>(ctx->inputs.at(0));
    auto path = value_cast<std::string>(ctx->inputs.at(1));
    std::ofstream fin(path);
    [[unlikely]] if (!fin)
        throw ztd::format_error("OSError: cannot open file for write: {}", path);
    writeMeshToOBJ(fin, *mesh);
}


ZENO_DOP_DEFUN(WriteOBJMesh, {}, {{
    "mesh", "save mesh to .obj file",
}, {
    {"mesh"},
    {"path"},
}, {
}});


}
}
ZENO_NAMESPACE_END
