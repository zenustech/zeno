#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/math/quaternion.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


std::variant<std::true_type, std::false_type> boolean_variant(bool cond) {
    if (cond) return std::true_type{};
    else return std::false_type{};
}


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = pointer_cast<Mesh>(ctx->inputs.at(0));
    auto translate = value_cast<math::vec3f>(ctx->inputs.at(1));
    auto scaling = value_cast<math::vec3f>(ctx->inputs.at(2));
    auto rotation = value_cast<math::vec4f>(ctx->inputs.at(3));
    auto rotmat = math::quaternion_matrix(rotation);

    zycl::default_queue().submit([=] (zycl::handler &cgh) {
        auto axr_vert = zycl::make_access<zycl::rwd>(cgh, mesh->vert);

        std::visit([&] (auto has_translate, auto has_scaling, auto has_rotation) {
            cgh.parallel_for(zycl::range<1>(mesh->vert.size()), [=] (zycl::item<1> it) {
                auto vert = axr_vert[it];
                if constexpr (has_scaling)
                    vert *= scaling;
                if constexpr (has_rotation)
                    vert = rotmat * vert;
                if constexpr (has_translate)
                    vert += translate;
                axr_vert[it] = vert;
            });
        }
        , boolean_variant(translate != math::vec3f(0))
        , boolean_variant(scaling != math::vec3f(1))
        , boolean_variant(rotation != math::vec4f(0, 0, 0, 1))
        );
    });

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(Mesh)});


}
}
ZENO_NAMESPACE_END
