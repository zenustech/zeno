#include <zeno/types/MeshTransform.h>
#include <zeno/math/quaternion.h>
#include <zeno/ztd/variant.h>
#include <tbb/parallel_for_each.h>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace types {


void transformMesh
    ( Mesh &mesh
    , math::vec3f const &translate
    , math::vec3f const &scaling
    , math::vec4f const &rotation
    ) {
    auto rotmat = math::quaternion_matrix(rotation);

    std::visit([&] (auto has_translate, auto has_scaling, auto has_rotation) {
#if 1
        tbb::parallel_for_each(begin(mesh.vert), end(mesh.vert), [&] (auto &vert) {
#else
        std::for_each(begin(mesh.vert), end(mesh.vert), [&] (auto &vert) {
#endif
            if constexpr (has_scaling)
                vert *= scaling;
            if constexpr (has_rotation)
                vert = rotmat * vert;
            if constexpr (has_translate)
                vert += translate;
        });
    }
    , ztd::make_bool_variant(translate != math::vec3f(0))
    , ztd::make_bool_variant(scaling != math::vec3f(1))
    , ztd::make_bool_variant(rotation != math::vec4f(0, 0, 0, 1))
    );
}


}
ZENO_NAMESPACE_END
