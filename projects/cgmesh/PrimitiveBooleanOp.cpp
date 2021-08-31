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
            // TODO: sort face order to prevent halfedge error:
            mesh.add_vertex_to_facet(f[0]);
            mesh.add_vertex_to_facet(f[1]);
            mesh.add_vertex_to_facet(f[2]);
            mesh.end_facet();
        }
        mesh.end_surface();
    }

};


auto prim_to_nef(PrimitiveObject *prim) {
    int nhalfedges = 10000;
    Polyhedron P(prim->size(), nhalfedges, prim->tris.size());
    MeshCreator<HalfedgeDS> meshCreator(prim->attr<vec3f>("pos"), prim->tris, nhalfedges);
    P.delegate(meshCreator);
    Nef_polyhedron nef(P);
    return nef;
}


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto prim1 = get_input<PrimitiveObject>("prim1");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto op_type = get_param<std::string>("op_type");

        auto nef1 = prim_to_nef(prim1.get());
        auto nef2 = prim_to_nef(prim2.get());

        Nef_polyhedron nef3;
        if (op_type == "union") {
            nef3 = nef1 + nef2;
        } else if (op_type == "intersection") {
            nef3 = nef1 * nef2;
        } else if (op_type == "difference") {
            nef3 = nef1 - nef2;
        } else {
            throw Exception("Bad boolean op: " + op_type);
        }
        Polyhedron result;
        nef3.convert_to_Polyhedron(result);

        auto res = std::make_shared<PrimitiveObject>();
        auto &pos = res->add_attr<vec3f>("pos");
        for (auto vit = result.vertices_begin(); vit != result.vertices_end(); vit++) {
            float x = vit->point().x().exact().convert_to<float>();
            float y = vit->point().y().exact().convert_to<float>();
            float z = vit->point().z().exact().convert_to<float>();
            pos.emplace_back(x, y, z);
        }
        res->resize(pos.size());
        for (auto fsit = result.facets_begin(); fsit != result.facets_end(); fsit++) {
            int fitid = 0;
            for (auto fit = fsit->facet_begin(); fitid++ < fsit->facet_degree(); fit++) {
                printf("%zd\n", fit->vertex_degree());
            }
        }

        set_output("prim", std::move(res));
    }
};

ZENO_DEFNODE(PrimitiveBooleanOp)({
    {
    "prim1", "prim2"
    },
    {
    "prim",
    },
    {
    {"enum union intersection difference", "op_type", "union"},
    },
    {"cgmesh"},
});

}
