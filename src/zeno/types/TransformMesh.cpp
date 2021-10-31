#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/math/quaternion.h>
#include <zeno/ztd/variant.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = (std::shared_ptr<Mesh>)ctx->inputs.at(0);
    auto translate = ctx->inputs.at(1).cast<math::vec3f>().value_or(math::vec3f(0));
    auto scaling = ctx->inputs.at(2).cast<math::vec3f>().value_or(math::vec3f(1));
    auto rotation = ctx->inputs.at(3).cast<math::vec4f>().value_or(math::vec4f(0, 0, 0, 1));
    auto rotmat = math::quaternion_matrix(rotation);

    zycl::queue().submit([&] (zycl::handler &cgh) {
        auto axr_vert = make_access<zycl::access::mode::discard_read_write>(cgh, mesh->vert);

        std::visit(
        [&] (auto translate, auto scaling, auto rotation) {
            cgh.parallel_for(zycl::range<1>(mesh->vert.size()), [=] (zycl::item<1> idx) {
                auto vert = axr_vert[idx];
                if constexpr (ztd::not_monostate<decltype(scaling)>)
                    vert *= scaling;
                if constexpr (ztd::not_monostate<decltype(rotation)>)
                    vert = rotation * vert;
                if constexpr (ztd::not_monostate<decltype(translate)>)
                    vert += translate;
                axr_vert[idx] = vert;
            });
        }
        , ztd::make_monovariant(vany(translate != math::vec3f(0)), translate)
        , ztd::make_monovariant(vany(scaling != math::vec3f(1)), scaling)
        , ztd::make_monovariant(vany(rotation != math::vec4f(0, 0, 0, 1)), rotmat)
        );
    });

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(std::shared_ptr<Mesh>)});


}
}
ZENO_NAMESPACE_END
