#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = (std::shared_ptr<Mesh>)ctx->inputs.at(0);
    auto translate = (ztd::vec3f)ctx->inputs.at(1);

    zycl::queue().submit([&] (zycl::handler &cgh) {
        auto axr_vert = make_access<zycl::access::mode::discard_read_write>(cgh, mesh->vert);

        cgh.parallel_for(zycl::range<1>(mesh->vert.size()), [=] (zycl::item<1> idx) {
            auto vert = axr_vert[idx];
            vert += translate;
            axr_vert[idx] = vert;
        });
    });

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(std::shared_ptr<Mesh>)});


}
}
ZENO_NAMESPACE_END
