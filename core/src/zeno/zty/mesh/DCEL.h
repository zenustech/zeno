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
        float co[3];
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
};



}
ZENO_NAMESPACE_END
