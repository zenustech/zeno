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


static void writeMeshToOBJ(std::ostream &out, Mesh &mesh) {
    decltype(auto) vert = mesh.vert.as_vector();
    decltype(auto) loop = mesh.loop.as_vector();
    decltype(auto) poly = mesh.poly.as_vector();

    // Write vertices
    for (auto &v : vert) {
        out << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }

    // Write indices
    for (auto &p : poly) {
        out << "f ";
        for (int l = p[0]; l < p[0] + p[1]; ++l) {
            out << loop[l] << " ";
        }
        out << "\n";
    }
}


static void WriteOBJMesh(dop::FuncContext *ctx) {
    auto mesh = ztd::cast<Mesh>(ctx->inputs.at(0));
    auto path = *ztd::cast<std::string>(ctx->inputs.at(1));
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
