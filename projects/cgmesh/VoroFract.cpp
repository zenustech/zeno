#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <voro++/voro++.hh>
#include <zinc/random.h>
#include <zinc/vec.h>
#include <vector>
#include <tuple>

namespace {

template <class T, class F>
void vorosplit(T const &particles, F const &factory) {
	int nx, ny, nz;
	double x, y, z;
    voro::voronoicell_neighbor c;
    std::vector<int> neigh, f_vert;
    std::vector<double> v;

    voro::pre_container pcon(-3,3,-3,3,0,6,false,false,false);
    /*for (int i = 0; i < particles.size(); i++) {
        auto p = particles[i];
        pcon.put(i + 1, p[0], p[1], p[2]);
    }*/
    pcon.import("/home/bate/Codes/zeno-blender/external/zeno/projects/cgmesh/pack_six_cube");
	pcon.guess_optimal(nx,ny,nz);

    voro::container con(-3,3,-3,3,0,6,nx,ny,nz,false,false,false,8);
	pcon.setup(con);

    voro::c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		cl.pos(x, y, z);

		c.neighbors(neigh);
		c.face_vertices(f_vert);
		c.vertices(x, y, z, v);

        auto &mesh = factory(false);

        for (int i = 0; i < (int)v.size(); i += 3) {
            mesh.verts().emplace_back(v[i], v[i+1], v[i+2]);
        }

		for(int i = 0, j = 0; i < (int)neigh.size(); i++) {
            int len = f_vert[j];
            int start = (int)mesh.loops().size();
            for (int k = j + 1; k < j + 1 + len; k++) {
                mesh.loops().push_back(f_vert[k]);
            }
            mesh.polys().emplace_back(start, len);
            j = j + 1 + len;
        }

	} while (cl.inc());
}

struct VoronoiFracture : zeno::INode {
    virtual void apply() override {
        auto boundaries = std::make_shared<zeno::ListObject>();
        auto interiors = std::make_shared<zeno::ListObject>();
        auto triangulate = get_param<bool>("triangulate");

        auto mesh = get_input<zeno::PrimitiveObject>("meshPrim");
        auto particles = get_input<zeno::PrimitiveObject>("particlesPrim");

        auto factory = [&] (auto isBoundary) -> decltype(auto) {
            auto ptr = std::make_shared<zeno::PrimitiveObject>();
            auto raw_ptr = ptr.get();
            if (isBoundary) {
                boundaries->arr.push_back(std::move(ptr));
            } else {
                interiors->arr.push_back(std::move(ptr));
            }
            return *raw_ptr;
        };
        vorosplit(particles->verts(), factory);

        if (triangulate) {
            for (auto const &mesh: boundaries->arr) {
                auto prim = zeno::smart_any_cast<std::shared_ptr<zeno::PrimitiveObject>>(mesh).get();
                prim_triangulate(prim);
            }
            for (auto const &mesh: interiors->arr) {
                auto prim = zeno::smart_any_cast<std::shared_ptr<zeno::PrimitiveObject>>(mesh).get();
                prim_triangulate(prim);
            }
        }

        set_output("boundaryPrimList", std::move(boundaries));
        set_output("interiorPrimList", std::move(interiors));
    }
};

ZENO_DEFNODE(VoronoiFracture)({
        { // inputs:
        {"PrimitiveObject", "meshPrim"},
        {"PrimitiveObject", "particlesPrim"},
        },
        { // outputs:
        {"ListObject", "interiorPrimList"},
        {"ListObject", "boundaryPrimList"},
        },
        {{"bool", "triangulate", "1"}},
        {"cgmesh"},
});

}
