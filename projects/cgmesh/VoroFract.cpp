#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/random.h>
#include <zeno/utils/vec.h>
#include <voro++/voro++.hh>
#include <vector>
#include <tuple>

namespace zeno {

struct VoronoiFracture : INode {
    virtual void apply() override {
        auto boundaries = std::make_shared<ListObject>();
        auto interiors = std::make_shared<ListObject>();
        auto triangulate = get_param<bool>("triangulate");

        auto mesh = get_input<PrimitiveObject>("meshPrim");

        {
            int nx, ny, nz;
            double x, y, z;
            voro::voronoicell_neighbor c;
            std::vector<int> neigh, f_vert;
            std::vector<double> v;

            voro::pre_container pcon(-1,1,-1,1,-1,1,false,false,false);

            if (has_input("particlesPrim")) {
                auto particlesPrim = get_input<PrimitiveObject>("particlesPrim");
                auto &parspos = particlesPrim->attr<vec3f>("pos");
                for (int i = 0; i < parspos.size(); i++) {
                    auto p = parspos[i];
                    pcon.put(i + 1, p[0], p[1], p[2]);
                }
            } else {
                auto numParticles = get_param<int>("numRandPoints");
                for (int i = 0; i < numParticles; i++) {
                    pcon.put(i + 1, frand()*2-1, frand()*2-1, frand()*2-1);
                }
            }

            pcon.guess_optimal(nx,ny,nz);
            voro::container con(-1,1,-1,1,-1,1,nx,ny,nz,false,false,false,8);
            pcon.setup(con);

            voro::c_loop_all cl(con);
            if(cl.start()) do if(con.compute_cell(c, cl)) {
                cl.pos(x, y, z);

                c.neighbors(neigh);
                c.face_vertices(f_vert);
                c.vertices(x, y, z, v);

                auto prim = std::make_shared<PrimitiveObject>();

                auto &pos = prim->add_attr<vec3f>("pos");
                for (int i = 0; i < (int)v.size(); i += 3) {
                    pos.emplace_back(v[i], v[i+1], v[i+2]);
                }
                prim->resize(pos.size());

                bool isBoundary = false;
                for (int i = 0, j = 0; i < (int)neigh.size(); i++) {
                    if (neigh[i] <= 0)
                        isBoundary = true;
                    if (neigh[i] == 0) printf("%d!!!\n", neigh[i]);
                    int len = f_vert[j];
                    int start = (int)prim->loops.size();
                    for (int k = j + 1; k < j + 1 + len; k++) {
                        prim->loops.push_back(f_vert[k]);
                    }
                    prim->polys.emplace_back(start, len);
                    j = j + 1 + len;
                }

                if (isBoundary) {
                    boundaries->arr.push_back(std::move(prim));
                } else {
                    interiors->arr.push_back(std::move(prim));
                }

            } while (cl.inc());
        }

        if (triangulate) {
            for (auto const &mesh: boundaries->arr) {
                auto prim = smart_any_cast<std::shared_ptr<PrimitiveObject>>(mesh).get();
                prim_triangulate(prim);
            }
            for (auto const &mesh: interiors->arr) {
                auto prim = smart_any_cast<std::shared_ptr<PrimitiveObject>>(mesh).get();
                prim_triangulate(prim);
            }
        }

        printf("VoronoiFracture got %zd boundaries, %zd interiors\n",
                boundaries->arr.size(), interiors->arr.size());

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
        { // params:
        {"bool", "triangulate", "1"},
        {"int", "numRandPoints", "256"},
        },
        {"cgmesh"},
});

}
