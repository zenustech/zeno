#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/ztd/functional.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


template <class T>
auto make_monovariant_if(auto &&cond, T x) {
    std::variant<std::monostate, T> ret;
    if (cond) {
        ret = std::move(x);
    }
    return ret;
}


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = (std::shared_ptr<Mesh>)ctx->inputs.at(0);
    auto translate = ctx->inputs.at(1).cast<ztd::vec3f>().value_or(ztd::vec3f(0));
    auto scaling = ctx->inputs.at(2).cast<ztd::vec3f>().value_or(ztd::vec3f(1));

    zycl::queue().submit([&] (zycl::handler &cgh) {
        auto axr_vert = make_access<zycl::access::mode::discard_read_write>(cgh, mesh->vert);

        std::visit(ztd::match
        ( [&] (std::monostate, std::monostate) {
        }
        , [&] (auto translate, auto scaling) {
            cgh.parallel_for(zycl::range<1>(mesh->vert.size()), [=] (zycl::item<1> idx) {
                auto vert = axr_vert[idx];
                if constexpr (!std::is_same_v<std::monostate, decltype(scaling)>)
                    vert *= scaling;
                if constexpr (!std::is_same_v<std::monostate, decltype(translate)>)
                    vert += translate;
                axr_vert[idx] = vert;
            });
        }
        )
        , make_monovariant_if(vall(translate == ztd::vec3f(0)), translate)
        , make_monovariant_if(vall(scaling == ztd::vec3f(1)), scaling)
        );
    });

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(std::shared_ptr<Mesh>)});


}
}
ZENO_NAMESPACE_END
