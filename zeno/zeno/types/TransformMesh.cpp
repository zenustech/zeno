#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/ztd/variant.h>
#include <zeno/math/quaternion.h>
#include <tbb/parallel_for.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = pointer_cast<Mesh>(ctx->inputs.at(0));
    auto translate = value_cast<math::vec3f>(ctx->inputs.at(1));
    auto scaling = value_cast<math::vec3f>(ctx->inputs.at(2));
    auto rotation = value_cast<math::vec4f>(ctx->inputs.at(3));
    auto rotmat = math::quaternion_matrix(rotation);

    std::visit([&] (auto has_translate, auto has_scaling, auto has_rotation) {
        tbb::parallel_for((size_t)0, mesh->vert.size(), (size_t)1, [&] (size_t it) {
            auto vert = mesh->vert[it];
            if constexpr (has_scaling)
                vert *= scaling;
            if constexpr (has_rotation)
                vert = rotmat * vert;
            if constexpr (has_translate)
                vert += translate;
            mesh->vert[it] = vert;
        });
    }
    , ztd::make_bool_variant(translate != math::vec3f(0))
    , ztd::make_bool_variant(scaling != math::vec3f(1))
    , ztd::make_bool_variant(rotation != math::vec4f(0, 0, 0, 1))
    );

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(Mesh)});


}
}
ZENO_NAMESPACE_END
