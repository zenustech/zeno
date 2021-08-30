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

template <class F>
void vorosplit(F const &factory) {
	int nx, ny, nz;
	double x, y, z;
    voro::voronoicell_neighbor c;
    std::vector<int> neigh, f_vert;
    std::vector<double> v;

    voro::pre_container pcon(-3,3,-3,3,0,6,false,false,false);
    for (int i = 0; i < 32; i++) {
        auto x = zinc::frand<double>(i);
        auto y = zinc::frand<double>(i);
        auto z = zinc::frand<double>(i);
        pcon.put(i + 1, x, y, z);
    }
	pcon.guess_optimal(nx,ny,nz);

    voro::container con(-3,3,-3,3,0,6,nx,ny,nz,false,false,false,8);
	pcon.setup(con);

    voro::c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		cl.pos(x, y, z);

		c.neighbors(neigh);
		c.face_vertices(f_vert);
		c.vertices(x, y, z, v);

        auto &mesh = factory(true);

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
        }

	} while (cl.inc());
}

struct VoronoiFracture : zeno::INode {
    virtual void apply() override {
        auto boundaries = std::make_shared<zeno::ListObject>();
        auto interiors = std::make_shared<zeno::ListObject>();
        auto triangulate = get_param<bool>("triangulate");

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
        vorosplit(factory);

        if (triangulate) {
            for (auto const &mesh: boundaries->arr) {
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
        {"PrimitiveObject", "prim"},
        },
        { // outputs:
        {"ListObject", "interiorPrimList"},
        {"ListObject", "boundaryPrimList"},
        },
        {{"bool", "triangulate", "1"}},
        {"cgmesh"},
});

}
