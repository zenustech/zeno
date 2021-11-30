#include <zeno/dop/dop.h>
#include <zeno/zty/mesh/Mesh.h>
#include <sstream>


ZENO_NAMESPACE_BEGIN
namespace zty {
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


static void MakeExampleInt(dop::FuncContext *ctx) {
    auto val = ztd::make_any<int>(42);
    ctx->outputs.at(0) = std::move(val);
}


ZENO_DOP_DEFUN(MakeExampleInt, {}, {{
        "misc", "make an example integer for demo",
    }, {
    }, {
        {"val"},
    }});


}
}
ZENO_NAMESPACE_END
