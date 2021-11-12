#pragma once


#include <vector>
#include <zeno/math/vec.h>
#include <zeno/ztd/vector.h>
#include <zeno/ztd/any_vector.h>


ZENO_NAMESPACE_BEGIN
namespace types {


struct Mesh {
    // points
    ztd::any_vector m_vert;
    ztd::vector<ztd::any_vector> m_poly;

    inline std::vector<math::vec3f> const &vert() const {
        return m_vert.cast<math::vec3f>();
    }

    template <size_t I = 3>
    inline std::vector<math::vec<I, uint32_t>> &poly() {
        return m_poly[I].cast<math::vec<I, uint32_t>>();
    }

    template <size_t I = 3>
    inline std::vector<math::vec<I, uint32_t>> const &poly() const {
        return m_poly[I].cast<math::vec<I, uint32_t>>();
    }
};



}
ZENO_NAMESPACE_END
