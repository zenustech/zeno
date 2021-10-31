#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/Mesh.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {
namespace {


template <class T>
auto make_monovariant_if(auto &&cond, T x) {
    std::variant<std::monostate, T> ret;
    if (cond) ret = std::move(x);
    return ret;
}


template <class T>
concept not_monostate = !std::is_same_v<std::monostate, std::remove_cvref_t<T>>;


struct rotation_matrix {
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;

    constexpr rotation_matrix(ztd::vec4f const &q) {
        // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        auto s = 2 / dot(q, q);
        auto [qi, qj, qk, qr] = std::make_tuple(q[0], q[1], q[2], q[3]);
        m11 = 1 - s * (qj*qj + qk*qk);
        m12 = s * (qi*qj - qk*qr);
        m13 = s * (qi*qk + qj*qr);
        m21 = s * (qi*qj + qk*qr);
        m22 = 1 - s * (qi*qi + qk*qk);
        m23 = s * (qj*qk - qi*qr);
        m31 = s * (qi*qk - qj*qr);
        m32 = s * (qj*qk + qi*qr);
        m33 = 1 - s * (qi*qi + qj*qj);
    }

    constexpr rotation_matrix &operator*=(ztd::vec3f const &v) {
        auto [vi, vj, vk] = std::make_tuple(v[0], v[1], v[2]);
        m11 *= vi; m21 *= vi; m31 *= vi;
        m12 *= vj; m22 *= vj; m32 *= vj;
        m13 *= vk; m23 *= vk; m33 *= vk;
        return *this;
    }

    constexpr ztd::vec3f operator%(ztd::vec3f const &v) const {
        auto [vi, vj, vk] = std::make_tuple(v[0], v[1], v[2]);
        return {
            m11 * vi + m12 * vj + m13 * vk,
            m21 * vi + m22 * vj + m23 * vk,
            m31 * vi + m32 * vj + m33 * vk,
        };
    }
};


static void TransformMesh(dop::FuncContext *ctx) {
    auto mesh = (std::shared_ptr<Mesh>)ctx->inputs.at(0);
    auto translate = ctx->inputs.at(1).cast<ztd::vec3f>().value_or(ztd::vec3f(0));
    auto scaling = ctx->inputs.at(2).cast<ztd::vec3f>().value_or(ztd::vec3f(1));
    auto rotation = ctx->inputs.at(2).cast<ztd::vec4f>().value_or(ztd::vec4f(0, 0, 0, 1));

    zycl::queue().submit([&] (zycl::handler &cgh) {
        auto axr_vert = make_access<zycl::access::mode::discard_read_write>(cgh, mesh->vert);

        std::visit(
        [&] (auto translate, auto scaling, auto rotation) {
            if constexpr (not_monostate<decltype(scaling)> && not_monostate<decltype(rotation)>) {
                rotation *= scaling;
            }
            cgh.parallel_for(zycl::range<1>(mesh->vert.size()), [=] (zycl::item<1> idx) {
                auto vert = axr_vert[idx];
                if constexpr (not_monostate<decltype(scaling)> && !not_monostate<decltype(rotation)>)
                    vert *= scaling;
                if constexpr (not_monostate<decltype(rotation)>)
                    vert = rotation % vert;
                if constexpr (not_monostate<decltype(translate)>)
                    vert += translate;
                axr_vert[idx] = vert;
            });
        }
        , make_monovariant_if(vall(translate == ztd::vec3f(0)), translate)
        , make_monovariant_if(vall(scaling == ztd::vec3f(1)), scaling)
        , make_monovariant_if(vall(rotation == ztd::vec4f(0, 0, 0, 1)), rotation_matrix(rotation))
        );
    });

    ctx->outputs.at(0) = std::move(mesh);
}


ZENO_DOP_IMPLEMENT(Transform, TransformMesh, {typeid(std::shared_ptr<Mesh>)});


}
}
ZENO_NAMESPACE_END
