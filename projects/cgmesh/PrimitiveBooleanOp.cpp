#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

namespace {
    using namespace zeno;

typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef Polyhedron::HalfedgeDS HalfedgeDS;
typedef HalfedgeDS::Vertex Vertex;
typedef Vertex::Point Point;
typedef Polyhedron::Halfedge_handle Halfedge_handle;
typedef Polyhedron::Halfedge_iterator Halfedge_iterator;
typedef Polyhedron::size_type size_type;



///<summery>
/// Creation of a mesh
///</summery>
template<class HDS>
class MeshCreator : public CGAL::Modifier_base<HDS>
{

    typedef typename HDS::Vertex Vertex;
    typedef typename Vertex::Point Point;

private:
    int _numVertices;
    int _numFacets;
    int _numHalfedges;
    std::vector<vec3f> &_vertices;
    std::vector<vec3i> &_triangles;

public:
    MeshCreator(std::vector<vec3f> & vertices, std::vector<vec3i> & triangles, int numHalfedges) :
        _numHalfedges(numHalfedges), _vertices(vertices), _triangles(triangles)
    {
        _numVertices = vertices.size();
        _numFacets = triangles.size();
    }

    void operator()(HDS &hds)
    {
        CGAL::Polyhedron_incremental_builder_3<HDS> mesh(hds);
        mesh.begin_surface(_numVertices, _numFacets);
        for (size_t i = 0; i < _numVertices; ++i) {
            auto v = _vertices[i];
            mesh.add_vertex(Point(v[0], v[1], v[2]));
        }
        for (size_t i = 0; i < _numFacets; ++i)
        {
            auto f = _triangles[i];
            mesh.begin_facet();
            mesh.add_vertex_to_facet(f[0]);
            mesh.add_vertex_to_facet(f[1]);
            mesh.add_vertex_to_facet(f[2]);
            mesh.end_facet();
        }
        mesh.end_surface();
    }

};


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto prim1 = get_input<PrimitiveObject>("prim1");
        auto prim2 = get_input<PrimitiveObject>("prim2");

        int prim1halfedges = 24;
        int prim2halfedges = 24;

        Polyhedron P1(prim1->size(), prim1halfedges, prim1->tris.size());
        Polyhedron P2(prim2->size(), prim2halfedges, prim2->tris.size());

        MeshCreator<HalfedgeDS> meshCreator1(prim1->attr<vec3f>("pos"), prim1->tris, prim1halfedges);
        MeshCreator<HalfedgeDS> meshCreator2(prim2->attr<vec3f>("pos"), prim2->tris, prim2halfedges);
        P1.delegate(meshCreator1);
        P2.delegate(meshCreator2);

        bool isClosed    = P1.is_closed()        && (P2.is_closed());
        bool isValid     = P1.is_valid()         && (P2.is_valid());
        bool isTriangles = P1.is_pure_triangle() && (P2.is_pure_triangle());

        Nef_polyhedron nef1(P1);
        Nef_polyhedron nef2(P2);

        Nef_polyhedron nef3 = nef1 * nef2;
        Polyhedron result;
        nef3.convert_to_Polyhedron(result);

        //for (auto vit = result.vertices_begin(); vit != result.vertices_end(); vit++) {
        //}
    }
};

}
