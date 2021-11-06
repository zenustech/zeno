#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <string_view>
#include <sstream>
#include <fstream>
#include <cstring>
#include <tuple>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void MakeExampleMesh(dop::FuncContext *ctx) {
    auto path = value_cast<std::string>(ctx->inputs.at(0));
    auto mesh = std::make_shared<Mesh>();
    std::ifstream fin(path);
    [[unlikely]] if (!fin)
        throw ztd::format_error("OSError: cannot open file for read: {}", path);
    readMeshFromOBJ(fin, *mesh);
    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_DEFUN(MakeExampleMesh, {}, {{
    "mesh", "make an example mesh for demo",
}, {
    {"path"},
}, {
    {"mesh"},
}});


}
}
ZENO_NAMESPACE_END
