#pragma once


#include <zeno/math/vec.h>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct Mesh;


struct DCEL
{
    static constexpr uint32_t kInvalid = 0x7fffffff;

    struct Vert
    {
        math::vec3f co;
    };

    struct Edge
    {
        uint32_t origin = kInvalid;    // Vert
        uint32_t twin = kInvalid;      // Edge
        uint32_t next = kInvalid;      // Edge
        uint32_t face = kInvalid;      // Face
    };

    struct Face
    {
        uint32_t first = kInvalid;     // Vert
    };

    std::vector<Vert> vert;
    std::vector<Edge> edge;
    std::vector<Face> face;

    DCEL() noexcept;
    DCEL(DCEL const &that);
    DCEL(DCEL &&that) noexcept;
    DCEL &operator=(DCEL const &that);
    DCEL &operator=(DCEL &&that) noexcept;

    explicit DCEL(Mesh const &mesh);
    explicit operator Mesh() const;

    DCEL subdivision();
};



}
ZENO_NAMESPACE_END
