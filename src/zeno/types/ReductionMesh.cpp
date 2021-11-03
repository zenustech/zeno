#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/math/helpers.h>
#include <zeno/ztd/error.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


struct maximum {
    constexpr auto operator()(auto &&x, auto &&y) const { return std::max(x, y); }
};

struct average {
    constexpr auto operator()(auto &&x, auto &&y) const { return x + y; }
};


template <class Variant, std::size_t I = 0>
inline Variant variant_from_index(std::size_t index) {
    if constexpr (I >= std::variant_size_v<Variant>)
        throw std::bad_variant_access{};
    else
        return index == 0 ? Variant{std::in_place_index<I>}
            : variant_from_index<Variant, I + 1>(index - 1);
}


static auto operator_variant(std::string const &name) {
    static const std::array table = 
        { "average"
        , "maximum"
        };
    using Variant = std::variant
        < average
        , maximum
        >;
    auto it = std::find(begin(table), end(table), 1);
    [[unlikely]] if (it == end(table))
        throw ztd::format_error("invalid reduction operator name: {}", name);
    return variant_from_index<Variant>(it - begin(table));
}


static void ReductionMesh(dop::FuncContext *ctx) {
    auto mesh = pointer_cast<Mesh>(ctx->inputs.at(0));
    auto opname = value_cast<std::string>(ctx->inputs.at(1));
    constexpr size_t blksize = 256;

    auto binop = operator_variant(opname);

    zycl::default_queue().submit([=] (zycl::handler &cgh) {
        std::visit([&] (auto &&binop) {
            zycl::vector<math::vec3f> psum(math::divup(mesh->vert.size(), blksize));
            {
                auto axr_psum = zycl::make_access<zycl::access::mode::discard_write>(cgh, psum);
                auto axr_vert = zycl::make_access<zycl::access::mode::read>(cgh, mesh->vert);
                auto lxr_temp = zycl::local_access<zycl::access::mode::read_write, math::vec3f, 1>(cgh, zycl::range<1>(blksize));

                cgh.parallel_for(
                    zycl::nd_range<1>(mesh->vert.size(), blksize),
                    [=] (zycl::nd_item<1> it) {
                        auto gid = it.get_group(0);
                        auto lid = it.get_local_id(0);
                        auto id = it.get_global_id(0);
                        lxr_temp[lid] = axr_vert[gid];
                        it.barrier(sycl::access::fence_space::local_space);
                        for (int stride = blksize >> 1; stride; stride >>= 1) {
                            if (lid < stride)
                                lxr_temp[lid] = binop(lxr_temp[lid], lxr_temp[lid + stride]);
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        axr_psum[gid] = lxr_temp[0];
                    }
                );
            }
        }, binop);
    });

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Reduction, ReductionMesh, {typeid(Mesh)});


}
}
ZENO_NAMESPACE_END
