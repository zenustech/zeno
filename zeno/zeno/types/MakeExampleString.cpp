#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/MeshIO.h>
#include <sstream>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void MakeExampleString(dop::FuncContext *ctx) {
    auto str = ztd::make_any<std::string>("models/Pig_Head.obj");
    ctx->outputs.at(0) = std::move(str);
}


ZENO_DOP_DEFUN(MakeExampleString, {}, {{
    "misc", "make an example string for demo",
}, {
}, {
    {"str"},
}});


}
}
ZENO_NAMESPACE_END
