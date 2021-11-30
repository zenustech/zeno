#include <zeno/dop/dop.h>
#include <zeno/zty/mesh/Mesh.h>
#include <zeno/zty/mesh/MeshTransform.h>


ZENO_NAMESPACE_BEGIN
namespace zty {
namespace {


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = pointer_cast<Mesh>(ctx->inputs.at(0));
    auto translate = value_cast<math::vec3f>(ctx->inputs.at(1));
    auto scaling = value_cast<math::vec3f>(ctx->inputs.at(2));
    auto rotation = value_cast<math::vec4f>(ctx->inputs.at(3));
    transformMesh(*mesh, translate, scaling, rotation);
    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(Mesh)});


}
}
ZENO_NAMESPACE_END
