#pragma once


#include <zeno/math/vec.h>
#include <list>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct Mesh;


struct DCEL
{
    struct Vert;
    struct Edge;
    struct Face;

    struct Vert
    {
        Edge *leaving;
        math::vec3f co;
    };

    struct Edge
    {
        Vert *origin;
        Edge *twin;
        Edge *next;
        Face *face;
    };

    struct Face
    {
        Edge *first;
    };

    std::list<Vert> vert;
    std::list<Edge> edge;
    std::list<Face> face;

    DCEL() noexcept;
    DCEL(DCEL const &that);
    DCEL(DCEL &&that) noexcept;
    DCEL &operator=(DCEL const &that);
    DCEL &operator=(DCEL &&that) noexcept;

    explicit DCEL(Mesh const &mesh);
    explicit operator Mesh() const;

    void subdivision();
};



}
ZENO_NAMESPACE_END
