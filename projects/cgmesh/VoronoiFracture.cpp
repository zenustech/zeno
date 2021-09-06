#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/random.h>
#include <zeno/utils/vec.h>
#include <voro++/voro++.hh>
#include <vector>
#include <tuple>

namespace {
using namespace zeno;

struct VoronoiFracture : INode {
    virtual void apply() override {
        auto boundaries = std::make_shared<ListObject>();
        auto interiors = std::make_shared<ListObject>();
        auto triangulate = get_param<bool>("triangulate");

        auto bmin = has_input("bboxMin") ?
            get_input<NumericObject>("bboxMin")->get<vec3f>() : vec3f(-1);
        auto bmax = has_axput("bboxMax") ?
            get_input<NumericObject>("bboxMax")->get<vec3f>() : vec3f(1);
        auto minx = bmin[0];
        auto miny = bmin[1];
        auto minz = bmin[2];
        auto maxx = bmax[0];
        auto maxy = bmax[1];
        auto maxz = bmax[2];
        auto periX = get_param<bool>("periodicX");
        auto periY = get_param<bool>("periodicY");
        auto periZ = get_param<bool>("periodicZ");

        {
            voro::pre_container pcon(minx,maxx,miny,maxy,minz,maxz,periX,periY,periZ);

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
                    vec3f p(frand(),frand(),frand());
                    p = p * (bmax - bmin) + bmin;
                    pcon.put(i + 1, p[0], p[1], p[2]);
                }
            }

            int nx, ny, nz;
            pcon.guess_optimal(nx,ny,nz);
            voro::container con(minx,maxx,miny,maxy,minz,maxz,nx,ny,nz,periX,periY,periZ,8);

            /*if (has_input("meshPrim")) {
                auto mesh = get_input<PrimitiveObject>("meshPrim");
                auto meshpos = mesh->attr<zeno::vec3f>("pos");
                for (int i = 0; i < mesh->tris.size(); i++) {
                    auto p = mesh->tris[i];
                    auto n = cross(
                                meshpos[p[0]] - meshpos[p[1]],
                                meshpos[p[0]] - meshpos[p[2]]);
                    n *= 1 / (length(n) + 1e-6);
                    auto c = dot(meshpos[p[0]], n);
                    printf("%f %f %f %f\n", n[0], n[1], n[2], c);
                    voro::wall_plane wal(n[0], n[1], n[2], c);
                    con.add_wall(wal);
                }
            }*/
            pcon.setup(con);

            voro::c_loop_all cl(con);
            voro::voronoicell_neighbor c;
            if(cl.start()) do if(con.compute_cell(c, cl)) {
                double x, y, z;
                cl.pos(x, y, z);

                std::vector<int> neigh, f_vert;
                c.neighbors(neigh);
                c.face_vertices(f_vert);
                std::vector<double> v;
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

        printf("VoronoiFracture got %zd boundaries, %zd interiors\n",
                boundaries->arr.size(), interiors->arr.size());

        if (triangulate) {
            for (auto const &prim: boundaries->get<std::shared_ptr<PrimitiveObject>>()) {
                prim_triangulate(prim.get());
            }
            for (auto const &prim: interiors->get<std::shared_ptr<PrimitiveObject>>()) {
                prim_triangulate(prim.get());
            }
        }

        set_output("primList", std::move(interiors));
        set_output("boundaryPrimList", std::move(boundaries));
    }
};

ZENO_DEFNODE(VoronoiFracture)({
        { // inputs:
        //{"PrimitiveObject", "meshPrim"},
        {"PrimitiveObject", "particlesPrim"},
        {"vec3f", "bboxMin", "-1,-1,-1"},
        {"vec3f", "bboxMax", "1,1,1"},
        },
        { // outputs:
        {"ListObject", "primList"},
        {"ListObject", "boundaryPrimList"},
        },
        { // params:
        {"bool", "triangulate", "1"},
        {"int", "numRandPoints", "256"},
        {"bool", "periodicX", "0"},
        {"bool", "periodicY", "0"},
        {"bool", "periodicZ", "0"},
        },
        {"cgmesh"},
});

}
