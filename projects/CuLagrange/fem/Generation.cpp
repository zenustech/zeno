#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/DihedralAngle.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"
#include <atomic>
#include <limits>
#include <type_traits>
#include <zeno/VDBGrid.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct ReadVtkMesh : INode {
    void apply() override {
        auto path = get_input<StringObject>("path")->get();
        auto prim = std::make_shared<PrimitiveObject>();
        auto &pos = prim->attr<vec3f>("pos");
        auto &quads = prim->quads;
        auto ompExec = zs::omp_exec();

        zs::Mesh<float, 3, int, 4> tet;
        read_tet_mesh_vtk(path, tet);
        const auto numVerts = tet.nodes.size();
        const auto numEles = tet.elems.size();
        prim->resize(numVerts);
        quads.resize(numEles);
        ompExec(zs::range(numVerts), [&](int i) { pos[i] = tet.nodes[i]; });
        ompExec(zs::range(numEles), [&](int i) { quads[i] = tet.elems[i]; });

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadVtkMesh, {/* inputs: */ {
                             {"readpath", "path"},
                         },
                         /* outputs: */
                         {
                             {"primitive", "prim"},
                         },
                         /* params: */
                         {},
                         /* category: */
                         {
                             "primitive",
                         }});

struct ExtractMeshSurface : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");
        auto &quads = prim->quads;
        auto ompExec = zs::omp_exec();
        const auto numVerts = pos.size();
        const auto numEles = quads.size();

        auto op = get_param<std::string>("op");
        bool includePoints = false;
        bool includeLines = false;
        bool includeTris = false;
        if (op == "all") {
            includePoints = true;
            includeLines = true;
            includeTris = true;
        } else if (op == "point")
            includePoints = true;
        else if (op == "edge")
            includeLines = true;
        else if (op == "surface")
            includeTris = true;

        std::vector<int> points;
        std::vector<float> pointAreas;
        std::vector<vec2i> lines;
        std::vector<float> lineAreas;
        std::vector<vec3i> tris;
#if 0
        {
            using namespace zs;
            zs::HashTable<int, 3, int> surfTable{0};
            constexpr auto space = zs::execspace_e::openmp;

            surfTable.resize(ompExec, 4 * numEles);
            surfTable.reset(ompExec, true);
            // compute getsurface
            // std::vector<int> tri2tet(4 * numEles);
            ompExec(range(numEles), [table = proxy<space>(surfTable), &quads](int ei) mutable {
                using table_t = RM_CVREF_T(table);
                using vec3i = zs::vec<int, 3>;
                auto record = [&table, ei](const vec3i &triInds) mutable {
                    if (auto sno = table.insert(triInds); sno != table_t::sentinel_v)
                        ; // tri2tet[sno] = ei;
                    else
                        printf("ridiculous, more than one tet share the same "
                               "surface!");
                };
                auto inds = quads[ei];
                record(vec3i{inds[0], inds[2], inds[1]});
                record(vec3i{inds[0], inds[3], inds[2]});
                record(vec3i{inds[0], inds[1], inds[3]});
                record(vec3i{inds[1], inds[2], inds[3]});
            });
            //
            tris.resize(numEles * 4);
            Vector<int> surfCnt{1, memsrc_e::host};
            surfCnt.setVal(0);
            ompExec(range(surfTable.size()),
                    [table = proxy<space>(surfTable), surfCnt = surfCnt.data(), &tris](int i) mutable {
                        using vec3i = zs::vec<int, 3>;
                        auto triInds = table._activeKeys[i];
                        using table_t = RM_CVREF_T(table);
                        if (table.query(vec3i{triInds[2], triInds[1], triInds[0]}) == table_t::sentinel_v &&
                            table.query(vec3i{triInds[1], triInds[0], triInds[2]}) == table_t::sentinel_v &&
                            table.query(vec3i{triInds[0], triInds[2], triInds[1]}) == table_t::sentinel_v)
                            tris[atomic_add(exec_omp, surfCnt, 1)] = zeno::vec3i{triInds[0], triInds[1], triInds[2]};
                    });
            auto scnt = surfCnt.getVal();
            tris.resize(scnt);
            fmt::print("{} surfaces\n", scnt);

            // surface points
            HashTable<int, 1, int> vertTable{numVerts};
            HashTable<int, 2, int> edgeTable{3 * numEles};
            vertTable.reset(ompExec, true);
            edgeTable.reset(ompExec, true);
            ompExec(tris,
                    [vertTable = proxy<space>(vertTable), edgeTable = proxy<space>(edgeTable)](vec3i triInds) mutable {
                        using vec1i = zs::vec<int, 1>;
                        using vec2i = zs::vec<int, 2>;
                        for (int d = 0; d != 3; ++d) {
                            vertTable.insert(vec1i{triInds[d]});
                            edgeTable.insert(vec2i{triInds[d], triInds[(d + 1) % 3]});
                        }
                    });
            auto svcnt = vertTable.size();
            points.resize(svcnt);
            pointAreas.resize(svcnt, 0.f);
            copy(mem_host, points.data(), vertTable._activeKeys.data(), sizeof(int) * svcnt);
            fmt::print("{} surface verts\n", svcnt);

            // surface edges
            Vector<int> surfEdgeCnt{1};
            surfEdgeCnt.setVal(0);
            auto dupEdgeCnt = edgeTable.size();
            std::vector<int> dupEdgeToSurfEdge(dupEdgeCnt, -1);
            lines.resize(dupEdgeCnt);
            ompExec(range(dupEdgeCnt), [edgeTable = proxy<space>(edgeTable), &lines, surfEdgeCnt = surfEdgeCnt.data(),
                                        &dupEdgeToSurfEdge](int edgeNo) mutable {
                using vec2i = zs::vec<int, 2>;
                vec2i edge = edgeTable._activeKeys[edgeNo];
                using table_t = RM_CVREF_T(edgeTable);
                if (auto eno = edgeTable.query(vec2i{edge[1], edge[0]});
                    eno == table_t::sentinel_v ||                        // opposite edge not exists
                    (eno != table_t::sentinel_v && edge[0] < edge[1])) { // opposite edge does exist
                    auto no = atomic_add(exec_omp, surfEdgeCnt, 1);
                    lines[no] = zeno::vec2i{edge[0], edge[1]};
                    dupEdgeToSurfEdge[edgeNo] = no;
                }
            });
            auto secnt = surfEdgeCnt.getVal();
            lines.resize(secnt);
            lineAreas.resize(secnt, 0.f);
            fmt::print("{} surface edges\n", secnt);

            ompExec(tris, [&, vertTable = proxy<space>(vertTable),
                           edgeTable = proxy<space>(edgeTable)](vec3i triInds) mutable {
                using vec3 = zs::vec<float, 3>;
                using vec1i = zs::vec<int, 1>;
                using vec2i = zs::vec<int, 2>;
                for (int d = 0; d != 3; ++d) {
                    auto p0 = vec3::from_array(pos[triInds[0]]);
                    auto p1 = vec3::from_array(pos[triInds[1]]);
                    auto p2 = vec3::from_array(pos[triInds[2]]);
                    float area = (p1 - p0).cross(p2 - p0).norm() / 2;
                    // surface vert
                    using vtable_t = RM_CVREF_T(vertTable);
                    auto vno = vertTable.query(vec1i{triInds[d]});
                    atomic_add(exec_omp, &pointAreas[vno], area / 3);
                    // surface edge
                    using etable_t = RM_CVREF_T(edgeTable);
#if 0
          auto eno = edgeTable.query(vec2i{triInds[d], triInds[(d + 1) % 3]});
          if (eno == etable_t::sentinel_v)
            continue;
          auto edge = edgeTable._activeKeys[eno];
          auto oEno = edgeTable.query(vec2i{triInds[(d + 1) % 3], triInds[d]});
          if ((edge[0] < edge[1] && oEno != etable_t::sentinel_v) ||
              oEno == etable_t::sentinel_v) {
            auto seNo = dupEdgeToSurfEdge[eno];
            atomic_add(exec_omp, &lineAreas[seNo], area / 3);
          }
#else
          auto eno = edgeTable.query(vec2i{triInds[(d + 1) % 3], triInds[d]});
          if (auto seNo = dupEdgeToSurfEdge[eno]; seNo != etable_t::sentinel_v)
            atomic_add(exec_omp, &lineAreas[seNo], area / 3);
#endif
                }
            });
        }
#else
        {
            /// surfaces
            auto comp_v3 = [](const vec3i &x, const vec3i &y) {
                for (int d = 0; d != 3; ++d) {
                    if (x[d] < y[d])
                        return 1;
                    else if (x[d] > y[d])
                        return 0;
                }
                return 0;
            };
            std::set<vec3i, RM_CVREF_T(comp_v3)> surfs(comp_v3);
            auto hastri = [&surfs](const vec3i &tri, int i, int j, int k) {
                return surfs.find(vec3i{tri[i], tri[j], tri[k]}) != surfs.end();
            };
            for (auto &&quad : quads) {
                surfs.insert(vec3i{quad[0], quad[2], quad[1]});
                surfs.insert(vec3i{quad[0], quad[3], quad[2]});
                surfs.insert(vec3i{quad[0], quad[1], quad[3]});
                surfs.insert(vec3i{quad[1], quad[2], quad[3]});
            }
            for (auto &&tri : surfs) {
                if (!hastri(tri, 2, 1, 0) && !hastri(tri, 1, 0, 2) && !hastri(tri, 0, 2, 1))
                    tris.push_back(vec3i{tri[0], tri[1], tri[2]});
            }

            /// surf edge
            auto comp_v2 = [](const vec2i &x, const vec2i &y) {
                return x[0] < y[0] ? 1 : (x[0] == y[0] && x[1] < y[1] ? 1 : 0);
            };
            std::set<vec2i, RM_CVREF_T(comp_v2)> sedges(comp_v2);
            auto ist2 = [&sedges, &lines](int i, int j) {
                if (sedges.find(vec2i{i, j}) == sedges.end() && sedges.find(vec2i{j, i}) == sedges.end()) {
                    sedges.insert(vec2i{i, j});
                    lines.push_back(vec2i{i, j});
                }
            };
            for (auto &&tri : tris) {
                ist2(tri[0], tri[1]);
                ist2(tri[1], tri[2]);
                ist2(tri[2], tri[0]);
            }
            lineAreas.resize(lines.size(), 0.f);

            /// surf verts
            std::set<int> spoints;
            auto ist = [&spoints, &points](int i) {
                if (spoints.find(i) == spoints.end()) {
                    spoints.insert(i);
                    points.push_back(i);
                }
            };
            for (auto &&line : lines) {
                ist(line[0]);
                ist(line[1]);
            }
            pointAreas.resize(points.size(), 0.f);
        }
#endif
        if (includeTris)
            prim->tris.values = tris; // surfaces
        if (includeLines) {
            prim->lines.values = lines; // surfaces edges
            prim->lines.add_attr<float>("area") = lineAreas;
        }
        if (includePoints) {
            prim->points.values = points; // surfaces points
            prim->points.add_attr<float>("area") = pointAreas;
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ExtractMeshSurface, {{{"quad (tet) mesh", "prim"}},
                                {{"mesh with surface topos", "prim"}},
                                {{"enum all point edge surface", "op", "all"}},
                                {"primitive"}});

struct ToBoundaryPrimitive : INode {
    void apply() override {
        using namespace zs;

        // base primitive
        auto inParticles = get_input<PrimitiveObject>("prim");
        auto &pos = inParticles->attr<vec3f>("pos");
        vec3f *velsPtr{nullptr};
        if (inParticles->has_attr("vel"))
            velsPtr = inParticles->attr<vec3f>("vel").data();
        auto &tris = inParticles->tris;
        std::size_t sprayedOffset = pos.size();

        //
        auto zsbou = std::make_shared<ZenoParticles>();

        // primitive binding
        zsbou->prim = inParticles;
        // set boundary flag
        zsbou->asBoundary = true;
        // sprayed offset
        zsbou->sprayedOffset = sprayedOffset;

        /// category, size
        std::size_t numVerts{pos.size()};
        std::size_t numEles{tris.size()};
        if (numEles == 0)
            throw std::runtime_error("boundary primitive is not a surface mesh!");

        ZenoParticles::category_e category{ZenoParticles::surface};

        // category
        zsbou->category = category;

        auto ompExec = zs::omp_exec();

        // attributes
        std::vector<zs::PropertyTag> tags{{"x", 3}, {"x0", 3}, {"v", 3}};
        std::vector<zs::PropertyTag> eleTags{{"inds", (int)3}};

        // verts
        zsbou->particles = std::make_shared<typename ZenoParticles::particles_t>(tags, numVerts, memsrc_e::host);
        auto &pars = zsbou->getParticles(); // tilevector
        ompExec(zs::range(numVerts), [pars = proxy<execspace_e::openmp>({}, pars), &pos, velsPtr](int pi) mutable {
            using vec3 = zs::vec<float, 3>;
            // pos
            pars.template tuple<3>("x", pi) = pos[pi];
            pars.template tuple<3>("x0", pi) = pos[pi];
            // vel
            if (velsPtr != nullptr)
                pars.template tuple<3>("v", pi) = velsPtr[pi];
            else
                pars.template tuple<3>("v", pi) = vec3::zeros();
        });

        // elements
        zsbou->elements = typename ZenoParticles::particles_t{eleTags, numEles, memsrc_e::host};
        auto &eles = zsbou->getQuadraturePoints(); // tilevector
        ompExec(zs::range(numEles), [pars = proxy<execspace_e::openmp>({}, pars),
                                     eles = proxy<execspace_e::openmp>({}, eles), &tris](size_t ei) mutable {
            // element-vertex indices
            // inds
            const auto &tri = tris[ei];
            for (int i = 0; i != 3; ++i)
                eles("inds", i, ei) = reinterpret_bits<float>(tri[i]);
        });

        /// extract surface edges
        if constexpr (true) {
            constexpr auto space = zs::execspace_e::openmp;

            using vec3i = zs::vec<int, 3>;
            using vec2i = zs::vec<int, 2>;
#if 0
      for (int i = 0; i != 10; ++i) {
        auto tri = tris[i];
        fmt::print("checking tri! {}-th tri<{}, {}, {}>\n", i, tri[0], tri[1], tri[2]);
        auto ii = tris.size() - 1 - i;
        tri = tris[ii];
        fmt::print("checking tri! {}-th tri<{}, {}, {}>\n", ii, tri[0], tri[1], tri[2]);
      }
#endif
#if 0
      zs::HashTable<int, 2, int> surfEdgeTable{3 * tris.size(), memsrc_e::host,
                                               -1};
      surfEdgeTable.resize(ompExec, 3 * tris.size());
      surfEdgeTable.reset(ompExec, true);
      ompExec(range(tris.size()),
              [&, seTable = proxy<space>(surfEdgeTable)](int ei) mutable {
                using table_t = RM_CVREF_T(seTable);
                auto tri = tris[ei];
                if (tri[0] == tri[1] || tri[0] == tri[2] || tri[1] == tri[2] ||
                    tri[0] < 0 || tri[1] < 0 || tri[2] < 0) {
                  fmt::print("what the fuck ? {}-th tri<{}, {}, {}>\n", ei,
                             tri[0], tri[1], tri[2]);
                }
                seTable.insert(vec2i{tri[0], tri[1]});
                seTable.insert(vec2i{tri[1], tri[2]});
                seTable.insert(vec2i{tri[2], tri[0]});
              });
      Vector<int> surfEdgeCnt{1, memsrc_e::host};
      surfEdgeCnt.setVal(0);
      auto &surfEdges = (*zsbou)[ZenoParticles::s_surfEdgeTag];
      surfEdges = typename ZenoParticles::particles_t({{ "inds",
                                                         2 }},
                                                      tris.size() * 3,
                                                      zs::memsrc_e::host);
      ompExec(range(surfEdgeTable.size()),
              [&, edges = proxy<space>({}, surfEdges),
               cnt = proxy<space>(surfEdgeCnt),
               seTable = proxy<space>(surfEdgeTable),
               n = surfEdgeTable.size()](int i) mutable {
                using table_t = RM_CVREF_T(seTable);
                auto edgeInds = seTable._activeKeys[i];
                if (auto no = seTable.query(vec2i{edgeInds[1], edgeInds[0]});
                    no == table_t::sentinel_v ||
                    (no != table_t::sentinel_v && edgeInds[0] < edgeInds[1])) {
                  auto id = atomic_add(exec_omp, &cnt[0], 1);
                  edges("inds", 0, id) = reinterpret_bits<float>(edgeInds[0]);
                  edges("inds", 1, id) = reinterpret_bits<float>(edgeInds[1]);
                }
              });
      auto seCnt = surfEdgeCnt.getVal();
      surfEdges.resize(seCnt);
      surfEdges = surfEdges.clone({zs::memsrc_e::device});
#else
            auto comp = [](const auto &x, const auto &y) {
                return x[0] < y[0] ? 1 : (x[0] == y[0] && x[1] < y[1] ? 1 : 0);
            };
            std::set<vec2i, RM_CVREF_T(comp)> sedges(comp);
            auto ist = [&sedges](int i, int j) {
                if (sedges.find(vec2i{i, j}) == sedges.end() && sedges.find(vec2i{j, i}) == sedges.end())
                    sedges.insert(vec2i{i, j});
            };
            for (auto &&tri : tris) {
                ist(tri[0], tri[1]);
                ist(tri[1], tri[2]);
                ist(tri[2], tri[0]);
            }
            auto &surfEdges = (*zsbou)[ZenoParticles::s_surfEdgeTag];
            surfEdges = typename ZenoParticles::particles_t({{"inds", 2}}, sedges.size(), zs::memsrc_e::host);
            int no = 0;
            auto sv = proxy<execspace_e::host>({}, surfEdges);
            for (auto &&edge : sedges) {
                sv("inds", 0, no) = reinterpret_bits<float>(edge[0]);
                sv("inds", 1, no) = reinterpret_bits<float>(edge[1]);
                no++;
            }
            surfEdges = surfEdges.clone({zs::memsrc_e::device});
#endif
            // surface vert indices
            auto &surfVerts = (*zsbou)[ZenoParticles::s_surfVertTag];
            surfVerts = typename ZenoParticles::particles_t({{"inds", 1}}, pos.size(), zs::memsrc_e::host);
            ompExec(zs::range(pos.size()), [&, surfVerts = proxy<space>({}, surfVerts)](int pointNo) mutable {
                surfVerts("inds", pointNo) = zs::reinterpret_bits<float>(pointNo);
            });
            // surface info
            surfVerts = surfVerts.clone({zs::memsrc_e::device});
        }

        eles = eles.clone({memsrc_e::device});
        pars = pars.clone({memsrc_e::device});

        set_output("ZSParticles", zsbou);
    }
};

ZENDEFNODE(ToBoundaryPrimitive, {
                                    {"prim"},
                                    {"ZSParticles"},
                                    {},
                                    {"FEM"},
                                });

struct ToZSTetrahedra : INode {
    void apply() override {
        using namespace zs;
        auto zsmodel = get_input<ZenoConstitutiveModel>("ZSModel");
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");
        auto &points = prim->points;
        auto &lines = prim->lines;
        auto &tris = prim->tris;
        auto &quads = prim->quads;

        bool include_customed_properties = get_param<int>("add_customed_attr");

        auto ompExec = zs::omp_exec();
        const auto numVerts = pos.size();
        const auto numEles = quads.size();

        auto zstets = std::make_shared<ZenoParticles>();
        zstets->prim = prim;
        zstets->getModel() = *zsmodel;
        zstets->category = ZenoParticles::tet;
        zstets->sprayedOffset = pos.size();

        std::vector<zs::PropertyTag> tags{
            {"m", 1},       {"x", 3},       {"x0", 3}, {"v", 3}, {"BCbasis", 9} /* normals for slip boundary*/,
            {"BCorder", 1}, {"BCtarget", 3}};
        std::vector<zs::PropertyTag> eleTags{{"vol", 1}, {"IB", 9}, {"inds", 4}, {"m", 1}};

        std::vector<zs::PropertyTag> auxVertAttribs{};
        std::vector<zs::PropertyTag> auxElmAttribs{};

        if (include_customed_properties) {
            for (auto &&[key, arr] : prim->verts.attrs) {
                const auto checkDuplication = [&tags](const std::string &name) {
                    for (std::size_t i = 0; i != tags.size(); ++i)
                        if (tags[i].name == name.data())
                            return true;
                    return false;
                };
                if (checkDuplication(key) || key == "pos" || key == "vel")
                    continue;
                const auto &k{key};
                match(
                    [&k, &auxVertAttribs](const std::vector<vec3f> &vals) {
                        auxVertAttribs.push_back(PropertyTag{k, 3});
                    },
                    [&k, &auxVertAttribs](const std::vector<float> &vals) {
                        auxVertAttribs.push_back(PropertyTag{k, 1});
                    },
                    [&k, &auxVertAttribs](const std::vector<vec3i> &vals) {},
                    [&k, &auxVertAttribs](const std::vector<int> &vals) {},
                    [](...) { throw std::runtime_error("what the heck is this type of attribute!"); })(arr);
            }

            for (auto &&[key, arr] : prim->quads.attrs) {
                const auto checkDuplication = [&eleTags](const std::string &name) {
                    for (std::size_t i = 0; i != eleTags.size(); ++i)
                        if (eleTags[i].name == name.data())
                            return true;
                    return false;
                };
                if (checkDuplication(key))
                    continue;
                const auto &k{key};
                match(
                    [&k, &auxElmAttribs](const std::vector<vec3f> &vals) {
                        auxElmAttribs.push_back(PropertyTag{k, 3});
                    },
                    [&k, &auxElmAttribs](const std::vector<float> &vals) {
                        auxElmAttribs.push_back(PropertyTag{k, 1});
                    },
                    [&k, &auxElmAttribs](const std::vector<vec3i> &vals) {},
                    [&k, &auxElmAttribs](const std::vector<int> &vals) {},
                    [](...) { throw std::runtime_error("what the heck is this type of attribute!"); })(arr);
            }
        }
        tags.insert(std::end(tags), std::begin(auxVertAttribs), std::end(auxVertAttribs));
        eleTags.insert(std::end(eleTags), std::begin(auxElmAttribs), std::end(auxElmAttribs));

        constexpr auto space = zs::execspace_e::openmp;
        zstets->particles = std::make_shared<typename ZenoParticles::particles_t>(tags, pos.size(), zs::memsrc_e::host);
        auto &pars = zstets->getParticles();
        // initialize the nodal attributes
        ompExec(zs::range(pos.size()), [&, pars = proxy<space>({}, pars)](int vi) mutable {
            using vec3 = zs::vec<float, 3>;
            using mat3 = zs::vec<float, 3, 3>;
            auto p = vec3::from_array(pos[vi]);
            pars.template tuple<3>("x", vi) = p;
            pars.template tuple<3>("x0", vi) = p;
            pars.template tuple<3>("v", vi) = vec3::zeros();
            if (prim->has_attr("vel"))
                pars.template tuple<3>("v", vi) = vec3::from_array(prim->attr<zeno::vec3f>("vel")[vi]);
            // default boundary handling setup
            pars.template tuple<9>("BCbasis", vi) = mat3::identity();
            pars("BCorder", vi) = 0;
            pars.template tuple<3>("BCtarget", vi) = vec3::zeros();
            // computed later
            pars("m", vi) = 0.f;

            for (auto &prop : auxVertAttribs) {
                if (prop.numChannels == 3)
                    pars.template tuple<3>(prop.name, vi) = prim->attr<vec3f>(std::string{prop.name})[vi];
                else // prop.numChannles == 1
                    pars(prop.name, vi) = prim->attr<float>(std::string{prop.name})[vi];
            }
        });
        zstets->elements = typename ZenoParticles::particles_t(eleTags, quads.size(), zs::memsrc_e::host);
        auto &eles = zstets->getQuadraturePoints();

        double volumeSum{0.0};
        // initialize element-wise attributes
        ompExec(zs::range(eles.size()),
                [&, pars = proxy<space>({}, pars), eles = proxy<space>({}, eles)](int ei) mutable {
                    using vec3 = zs::vec<float, 3>;
                    using mat3 = zs::vec<float, 3, 3>;
                    using vec4 = zs::vec<float, 4>;
                    auto quad = quads[ei];
                    vec3 xs[4];
                    for (int d = 0; d != 4; ++d) {
                        eles("inds", d, ei) = zs::reinterpret_bits<float>(quad[d]);
                        xs[d] = pars.template pack<3>("x", quad[d]);
                    }

                    vec3 ds[3] = {xs[1] - xs[0], xs[2] - xs[0], xs[3] - xs[0]};
                    mat3 D{};
                    for (int d = 0; d != 3; ++d)
                        for (int i = 0; i != 3; ++i)
                            D(d, i) = ds[i][d];
                    eles.template tuple<9>("IB", ei) = zs::inverse(D);
                    auto vol = zs::abs(zs::determinant(D)) / 6;
                    atomic_add(exec_omp, &volumeSum, (double)vol);
                    eles("vol", ei) = vol;
                    // vert masses
                    // auto vmass = vol * zsmodel->density / 4;
                    if(pars.hasProperty("phi")){
                        float phi = 0;
                        for(int i = 0;i != 4;++i)
                            phi += pars("phi",quad[i]);
                        phi /= 4.0;
                        eles("m",ei) = vol * phi;
                    }else
                        eles("m", ei) = vol * zsmodel->density;

                    auto vmass = eles("m",ei) / 4;
                    for (int d = 0; d != 4; ++d)
                        atomic_add(zs::exec_omp, &pars("m", quad[d]), vmass);

                    for (auto &prop : auxElmAttribs) {
                        if (prop.numChannels == 3)
                            eles.template tuple<3>(prop.name, ei) = prim->quads.attr<vec3f>(std::string{prop.name})[ei];
                        else
                            eles(prop.name, ei) = prim->quads.attr<float>(std::string{prop.name})[ei];
                    }
                });
        zstets->setMeta("meanMass", (float)(volumeSum * zsmodel->density / pars.size()));

        // surface info
        double areaSum{0.0};
        auto &surfaces = (*zstets)[ZenoParticles::s_surfTriTag];
        surfaces = typename ZenoParticles::particles_t({{"inds", 3}}, tris.size(), zs::memsrc_e::host);
        ompExec(zs::range(tris.size()),
                [&, surfaces = proxy<space>({}, surfaces), pars = proxy<space>({}, pars)](int triNo) mutable {
                    auto tri = tris[triNo];
                    auto X0 = pars.template pack<3>("x0", tri[0]);
                    auto X1 = pars.template pack<3>("x0", tri[1]);
                    auto X2 = pars.template pack<3>("x0", tri[2]);
                    atomic_add(exec_omp, &areaSum, (double)(X1 - X0).cross(X2 - X0).norm() / 2);
                    for (int i = 0; i != 3; ++i)
                        surfaces("inds", i, triNo) = zs::reinterpret_bits<float>(tri[i]);
                });

        ompExec(zs::range(tris.size()), [&, surfaces = proxy<space>({}, surfaces)](int triNo) mutable {
            // if(lineNo == 1) {
            auto check_tri = surfaces.pack(dim_c<3>, "inds", triNo, int_c);
            auto tri = tris[triNo];
            if (tri[0] != check_tri[0] || tri[1] != check_tri[1] || tri[2] != check_tri[2]) {
                printf("GENERATION::tri_mismatch%d : [%d %d %d] != [%d %d %d]\n", triNo, check_tri[0], check_tri[1],
                       check_tri[2], tri[0], tri[1], tri[2]);
            }
            // }
        });

        // record total surface area
        zstets->setMeta("surfArea", (float)areaSum);

        auto &surfEdges = (*zstets)[ZenoParticles::s_surfEdgeTag];
        surfEdges = typename ZenoParticles::particles_t({{"inds", 2}, {"w", 1}}, lines.size(), zs::memsrc_e::host);
        // const auto &lineAreas = lines.attr<float>("area");

        ompExec(zs::range(lines.size()), [&, surfEdges = proxy<space>({}, surfEdges)](int lineNo) mutable {
            auto line = lines[lineNo];
            for (int i = 0; i != 2; ++i) {
                // int32_t idx = line[i];
                // surfEdges("inds", i, lineNo) = zs::reinterpret_bits<float>(idx);
                surfEdges("inds", i, lineNo) = zs::reinterpret_bits<float>(line[i]);
            }

            if (lineNo == 0) {
                // auto check_edge = surfEdges.template pack<2>("inds",lineNo).template reinterpret_bits<int>();
                // printf("GENERATION_A::line0 : %d %d\n",line[0],line[1]);
            }

            // surfEdges("w", lineNo) = lineAreas[lineNo]; // line area (weight)
        });

        ompExec(zs::range(lines.size()), [&, surfEdges = proxy<space>({}, surfEdges)](int lineNo) mutable {
            // if(lineNo == 1) {
            auto check_edge = surfEdges.template pack<2>("inds", lineNo).template reinterpret_bits<int32_t>();
            auto line = lines[lineNo];
            if (line[0] != check_edge[0] || line[1] != check_edge[1]) {
                printf("GENERATION::line_mismatch%d : [%d %d] != [%d %d]\n", lineNo, check_edge[0], check_edge[1],
                       line[0], line[1]);
                // printf("REF::line%d : %d %d\n",lineNo,line[0],line[1]);
            }
            // }
        });

        auto &surfVerts = (*zstets)[ZenoParticles::s_surfVertTag];
        surfVerts = typename ZenoParticles::particles_t({{"inds", 1}, {"w", 1}}, points.size(), zs::memsrc_e::host);
        // const auto &pointAreas = points.attr<float>("area");
        ompExec(zs::range(points.size()), [&, surfVerts = proxy<space>({}, surfVerts)](int pointNo) mutable {
            auto point = points[pointNo];
            surfVerts("inds", pointNo) = zs::reinterpret_bits<float>(point);
            // surfVerts("w", pointNo) = pointAreas[pointNo]; // point area (weight)
        });

        pars = pars.clone({zs::memsrc_e::device});
        eles = eles.clone({zs::memsrc_e::device});
        surfaces = surfaces.clone({zs::memsrc_e::device});
        surfEdges = surfEdges.clone({zs::memsrc_e::device});
        surfVerts = surfVerts.clone({zs::memsrc_e::device});

        // auto cudaExec = cuda_exec();
        // constexpr auto cuda_space = zs::execspace_e::cuda;

        // cudaExec(range(lines.size()),
        //     [lines = proxy<cuda_space>({},surfEdges)] ZS_LAMBDA(int li) mutable {
        //         auto inds = lines.template pack<2>("inds",li).template reinterpret_bits<int>();
        //         if(li == 0)
        //             printf("line0 : %d %d\n",inds[0],inds[1]);
        // });

        set_output("ZSParticles", std::move(zstets));
    }
};

ZENDEFNODE(ToZSTetrahedra, {{{"ZSModel"}, {"quad (tet) mesh", "prim"}},
                            {{"tetmesh on gpu", "ZSParticles"}},
                            {{"int", "add_customed_attr", "0"}},
                            {"FEM"}});

struct ToZSTriMesh : INode {
    using T = float;
    using dtiles_t = zs::TileVector<T, 32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;

    constexpr T area(T a, T b, T c) {
        T s = (a + b + c) / 2;
        return zs::sqrt(s * (s - a) * (s - b) * (s - c));
    }

    void apply() override {
        using namespace zs;
        // auto zsmodel = get_input<ZenoConstitutiveModel>("ZSModel");
        auto prim = get_input<PrimitiveObject>("prim");
        const auto &pos = prim->attr<zeno::vec3f>("pos");
        zeno::vec3f *velsPtr = nullptr;
        if (prim->has_attr("vel"))
            velsPtr = prim->attr<zeno::vec3f>("vel").data();
        const auto &points = prim->points;
        const auto &lines = prim->lines;
        const auto &tris = prim->tris;

        auto ompExec = zs::omp_exec();
        const auto numVerts = pos.size();
        const auto numTris = tris.size();

        auto zstris = std::make_shared<ZenoParticles>();
        zstris->prim = prim;
        // zstris->getModel() = *zsmodel;
        zstris->category = ZenoParticles::surface;
        zstris->sprayedOffset = pos.size();

        bool include_customed_properties = get_param<int>("add_customed_attr");

        std::vector<zs::PropertyTag> tags{{"x", 3}, {"v", 3}, {"inds", 1}};
        std::vector<zs::PropertyTag> eleTags{{"inds", 3}, {"area", 1}};

        std::vector<zs::PropertyTag> auxVertAttribs{};
        std::vector<zs::PropertyTag> auxElmAttribs{};

        if (include_customed_properties) {
            for (auto &&[key, arr] : prim->verts.attrs) {
                const auto checkDuplication = [&tags](const std::string &name) {
                    for (std::size_t i = 0; i != tags.size(); ++i)
                        if (tags[i].name == name.data())
                            return true;
                    return false;
                };
                if (checkDuplication(key) || key == "pos" || key == "vel")
                    continue;
                const auto &k{key};
                match(
                    [&k, &auxVertAttribs](const std::vector<vec3f> &vals) {
                        auxVertAttribs.push_back(PropertyTag{k, 3});
                    },
                    [&k, &auxVertAttribs](const std::vector<float> &vals) {
                        auxVertAttribs.push_back(PropertyTag{k, 1});
                    },
                    [&k, &auxVertAttribs](const std::vector<vec3i> &vals) {},
                    [&k, &auxVertAttribs](const std::vector<int> &vals) {},
                    [](...) { throw std::runtime_error("what the heck is this type of attribute!"); })(arr);
            }
            for (auto &&[key, arr] : prim->tris.attrs) {
                const auto checkDuplication = [&eleTags](const std::string &name) {
                    for (std::size_t i = 0; i != eleTags.size(); ++i)
                        if (eleTags[i].name == name.data())
                            return true;
                    return false;
                };
                if (checkDuplication(key))
                    continue;
                const auto &k{key};
                match(
                    [&k, &auxElmAttribs](const std::vector<vec3f> &vals) {
                        auxElmAttribs.push_back(PropertyTag{k, 3});
                    },
                    [&k, &auxElmAttribs](const std::vector<float> &vals) {
                        auxElmAttribs.push_back(PropertyTag{k, 1});
                    },
                    [&k, &auxElmAttribs](const std::vector<vec3i> &vals) {},
                    [&k, &auxElmAttribs](const std::vector<int> &vals) {},
                    [](...) { throw std::runtime_error("what the heck is this type of attribute!"); })(arr);
            }
        }

        tags.insert(std::end(tags), std::begin(auxVertAttribs), std::end(auxVertAttribs));
        eleTags.insert(std::end(eleTags), std::begin(auxElmAttribs), std::end(auxElmAttribs));

        zstris->setMeta(ZenoParticles::s_userDataTag, prim->userData());

        constexpr auto space = zs::execspace_e::openmp;
        zstris->particles = std::make_shared<tiles_t>(tags, pos.size(), zs::memsrc_e::host);
        auto &pars = zstris->getParticles();
        ompExec(Collapse{pars.size()},
                [pars = proxy<space>({}, pars), &pos, prim, &auxVertAttribs, velsPtr](int vi) mutable {
                    pars.template tuple<3>("x", vi) = vec3::from_array(pos[vi]);
                    pars("inds", vi, int_c) = vi;
                    auto vel = vec3::zeros();
                    if (velsPtr != nullptr)
                        vel = vec3::from_array(velsPtr[vi]);
                    pars.template tuple<3>("v", vi) = vel;

                    for (auto &prop : auxVertAttribs) {
                        if (prop.numChannels == 3)
                            pars.template tuple<3>(prop.name, vi) = prim->attr<vec3f>(std::string{prop.name})[vi];
                        else // prop.numChannles == 1
                            pars(prop.name, vi) = prim->attr<float>(std::string{prop.name})[vi];
                    }
                });

        zstris->elements = typename ZenoParticles::particles_t(eleTags, tris.size(), zs::memsrc_e::host);
        auto &eles = zstris->getQuadraturePoints();
        ompExec(Collapse{tris.size()}, [this, eles = proxy<space>({}, eles), pars = proxy<space>({}, pars), &tris,
                                        &auxElmAttribs](int ei) mutable {
            T l[3] = {};
            for (size_t i = 0; i < 3; ++i) {
                eles("inds", i, ei) = zs::reinterpret_bits<float>(tris[ei][i]);
                l[i] = (pars.template pack<3>("x", tris[ei][i]) - pars.template pack<3>("x", tris[ei][(i + 1) % 3]))
                           .length();
            }
            eles("area", ei) = area(l[0], l[1], l[2]);

            for (auto &prop : auxElmAttribs) {
                if (prop.numChannels == 3)
                    eles.template tuple<3>(prop.name, ei) = tris.attr<vec3f>(std::string{prop.name})[ei];
                else
                    eles(prop.name, ei) = tris.attr<float>(std::string{prop.name})[ei];
            }
        });

        pars = pars.clone({zs::memsrc_e::device});
        eles = eles.clone({zs::memsrc_e::device});

        set_output("ZSParticles", std::move(zstris));
    }
};

ZENDEFNODE(ToZSTriMesh, {{{"surf (tri) mesh", "prim"}},
                         {{"trimesh on gpu", "ZSParticles"}},
                         {{"int", "add_customed_attr", "0"}},
                         {"FEM"}});

struct ToZSSurfaceMesh : INode {
    using T = float;
    using dtiles_t = typename ZenoParticles::dtiles_t;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;

    void apply() override {
        using namespace zs;
        auto zsmodel = get_input<ZenoConstitutiveModel>("ZSModel");
        auto prim = get_input<PrimitiveObject>("prim");
        bool useDouble = get_input2<bool>("high_precision");
        bool withBending = get_input2<bool>("with_bending");
        const auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &points = prim->points;
        const auto &lines = prim->lines;
        const auto &tris = prim->tris;

        auto ompExec = zs::omp_exec();
        const auto numVerts = pos.size();
        const auto numTris = tris.size();

        auto zstris = std::make_shared<ZenoParticles>();
        zstris->prim = prim;
        zstris->getModel() = *zsmodel;
        zstris->category = ZenoParticles::surface;
        zstris->sprayedOffset = pos.size();

        std::vector<zs::PropertyTag> tags{
            {"m", 1},       {"x", 3},       {"x0", 3},      {"v", 3}, {"BCbasis", 9} /* normals for slip boundary*/,
            {"BCorder", 1}, {"BCfixed", 1}, {"BCtarget", 3}};
        std::vector<zs::PropertyTag> eleTags{{"vol", 1}, {"IB", 4}, {"inds", 3}};

        constexpr auto space = zs::execspace_e::openmp;
        std::variant<std::true_type, std::false_type> tag;
        if (useDouble)
            tag = std::true_type{};
        else
            tag = std::false_type{};

        float scaling = 1.f;
        if (has_input("rest_shape_scaling")) {
            scaling = get_input2<float>("rest_shape_scaling");
            if (scaling < std::numeric_limits<float>::epsilon() * 10)
                scaling = 1.f;
        }

        match([&](auto tag) {
            using namespace zs;
            constexpr auto space = zs::execspace_e::openmp;
            constexpr bool use_double = RM_CVREF_T(tag)::value;
            using T = conditional_t<use_double, double, float>;

            float E, nu;
            match([&E, &nu](const auto &model) {
                auto [E_self, nu_self] = E_nu_from_lame_parameters(model.mu, model.lam);
                E = E_self;
                nu = nu_self;
            })(zsmodel->getElasticModel());

            if constexpr (use_double)
                zstris->getParticles<true>() = dtiles_t{tags, pos.size(), zs::memsrc_e::host};
            else
                zstris->particles = std::make_shared<tiles_t>(tags, pos.size(), zs::memsrc_e::host);
            auto &pars = zstris->getParticles<use_double>();
            ompExec(Collapse{pars.size()}, [pars = proxy<space>({}, pars), &pos, &prim](int vi) mutable {
                using mat3 = zs::vec<T, 3, 3>;
                auto p = vec_to_other<zs::vec<T, 3>>(pos[vi]);
                pars.tuple(dim_c<3>, "x", vi) = p;
                pars.tuple(dim_c<3>, "x0", vi) = p;
                pars.tuple(dim_c<3>, "v", vi) = zs::vec<T, 3>::zeros();
                if (prim->has_attr("vel"))
                    pars.tuple(dim_c<3>, "v", vi) = vec_to_other<zs::vec<T, 3>>(prim->attr<zeno::vec3f>("vel")[vi]);
                // default boundary handling setup
                pars.tuple(dim_c<3, 3>, "BCbasis", vi) = mat3::identity();
                pars("BCorder", vi) = 0;
                pars("BCfixed", vi) = 0;
                pars.tuple(dim_c<3>, "BCtarget", vi) = zs::vec<T, 3>::zeros();
                // computed later
                pars("m", vi) = 0;
            });

            zstris->elements = typename ZenoParticles::particles_t(eleTags, tris.size(), zs::memsrc_e::host);
            auto &eles = zstris->getQuadraturePoints();
            ompExec(Collapse{tris.size()}, [&zsmodel, pars = proxy<space>({}, pars), eles = proxy<space>({}, eles),
                                            &tris, scaling](int ei) mutable {
                auto tri = tris[ei];
                using vec3 = zs::vec<double, 3>;
                using mat2 = zs::vec<double, 2, 2>;
                vec3 xs[3];
                for (int d = 0; d != 3; ++d) {
                    eles("inds", d, ei, int_c) = tri[d];
                    xs[d] = pars.pack(dim_c<3>, "x", tri[d]);
                }

                vec3 ds[2] = {(xs[1] - xs[0]) * scaling, (xs[2] - xs[0]) * scaling};

                // ref: codim-ipc
                // for first fundamental form
                mat2 B{};
#if 0
              B(0, 0) = ds[0].l2NormSqr();
              B(1, 0) = B(0, 1) = ds[0].dot(ds[1]);
              B(1, 1) = ds[1].l2NormSqr();
#else
            B(0, 0) = ds[0].norm();
            B(1, 0) = 0;
            B(0, 1) = ds[0].dot(ds[1]) / B(0, 0);
            B(1, 1) = ds[0].cross(ds[1]).norm() / B(0, 0);
#endif
                auto IB = inverse(B);
                {
                    if (std::isnan(IB(0, 0)) || std::isnan(IB(0, 1)) || std::isnan(IB(1, 0)) || std::isnan(IB(1, 1))) {
#if 0
                        fmt::print(fg(fmt::color::light_golden_rod_yellow), "B[{}]: [{}, {}; {}, {}]\n", ei, B(0, 0),
                                   B(0, 1), B(1, 0), B(1, 1));
                        fmt::print(fg(fmt::color::light_sea_green), "IB[{}]: [{}, {}; {}, {}]\n", ei, IB(0, 0),
                                   IB(0, 1), IB(1, 0), IB(1, 1));
                        fmt::print(fg(fmt::color::yellow_green), "tri[{}]: <{}, {}, {}> - <{}, {}, {}> - \n", ei, IB(0, 0), IB(0, 1),
                                   IB(1, 0), IB(1, 1));
#else
                        IB = mat2::zeros();
#endif
                    }
                }
                eles.template tuple<4>("IB", ei) = IB;

                auto vol = ds[0].cross(ds[1]).norm() / 2 * zsmodel->dx;
                eles("vol", ei) = vol;
                // vert masses
                auto vmass = vol * zsmodel->density / 3;
                for (int d = 0; d != 3; ++d)
                    atomic_add(zs::exec_omp, &pars("m", tri[d]), (T)vmass);
            });

#if 0
      zs::HashTable<int, 2, int> surfEdgeTable{0};
      surfEdgeTable.resize(ompExec, 3 * tris.size());
      surfEdgeTable.reset(ompExec, true);

      auto seTable = proxy<space>(surfEdgeTable);
      using table_t = RM_CVREF_T(seTable);
      using vec3i = zs::vec<int, 3>;
      using vec2i = zs::vec<int, 2>;
      ompExec(range(tris.size()), [&](int ei) {
          auto tri = tris[ei];
          seTable.insert(vec2i{tri[0], tri[1]});
          seTable.insert(vec2i{tri[1], tri[2]});
          seTable.insert(vec2i{tri[2], tri[0]});
      });
      Vector<int> surfEdgeCnt{1, memsrc_e::host};
      surfEdgeCnt.setVal(0);
      auto &surfEdges = (*zstris)[ZenoParticles::s_surfEdgeTag];
      surfEdges = typename ZenoParticles::particles_t({{ "inds",
                                                         2 }},
                                                      tris.size() * 3,
                                                      zs::memsrc_e::host);
      ompExec(range(seTable.size()), [&, edges = proxy<space>({}, surfEdges),
                                      cnt = proxy<space>(surfEdgeCnt)](
                                         int i) mutable {
          auto edgeInds = seTable._activeKeys[i];
          if (auto no = seTable.query(vec2i{edgeInds[1], edgeInds[0]});
              no == table_t::sentinel_v ||
              (no != table_t::sentinel_v && edgeInds[0] < edgeInds[1])) {
              auto id = atomic_add(exec_omp, &cnt[0], 1);
              edges("inds", 0, id) = reinterpret_bits<float>(edgeInds[0]);
              edges("inds", 1, id) = reinterpret_bits<float>(edgeInds[1]);
          }
      });
      auto seCnt = surfEdgeCnt.getVal();
      surfEdges.resize(seCnt);
      surfEdges = surfEdges.clone({zs::memsrc_e::device});
#else
            auto comp = [](const auto &x, const auto &y) {
                return x[0] < y[0] ? 1 : (x[0] == y[0] && x[1] < y[1] ? 1 : 0);
            };
            std::map<vec2i, int, RM_CVREF_T(comp)> edge2tri(comp);
            for (int i = 0; i != tris.size(); ++i) {
                auto tri = tris[i];
                for (int k = 0; k != 3; ++k) {
                    auto e0 = tri[k];
                    auto e1 = tri[(k + 1) % 3];
                    if (withBending) {
                        if (edge2tri.find(vec2i{e0, e1}) != edge2tri.end())
                            throw std::runtime_error(
                                fmt::format("the same edge <{}, {}> is being shared by multiple triangles!", e0, e1));
                    }
                    edge2tri[vec2i{e0, e1}] = i;
                }
            }
            std::set<vec2i, RM_CVREF_T(comp)> sedges(comp);
            using vec4i = std::array<int, 4>;
            std::vector<vec4i> bedges;
            for (auto &&tri : tris) {
                {
                    int i = tri[0], j = tri[1];
                    auto it = sedges.find(vec2i{j, i});
                    if (sedges.find(vec2i{i, j}) == sedges.end() && it == sedges.end())
                        sedges.insert(vec2i{i, j});
                    else if (it != sedges.end()) {
                        auto neighborTriNo = edge2tri[vec2i{j, i}];
                        auto neighborTri = tris[neighborTriNo];

                        auto selfVertNo = tri[2];
                        int neighborVertNo = -1;
                        for (int k = 0; k != 3; ++k)
                            if (neighborTri[k] != i && neighborTri[k] != j) {
                                neighborVertNo = neighborTri[k];
                                break;
                            }
                        bedges.push_back(vec4i{selfVertNo, i, j, neighborVertNo});
                    }
                }
                {
                    int i = tri[1], j = tri[2];
                    auto it = sedges.find(vec2i{j, i});
                    if (sedges.find(vec2i{i, j}) == sedges.end() && it == sedges.end())
                        sedges.insert(vec2i{i, j});
                    else if (it != sedges.end()) {
                        auto neighborTriNo = edge2tri[vec2i{j, i}];
                        auto neighborTri = tris[neighborTriNo];

                        auto selfVertNo = tri[0];
                        int neighborVertNo = -1;
                        for (int k = 0; k != 3; ++k)
                            if (neighborTri[k] != i && neighborTri[k] != j) {
                                neighborVertNo = neighborTri[k];
                                break;
                            }
                        bedges.push_back(vec4i{selfVertNo, i, j, neighborVertNo});
                    }
                }
                {
                    int i = tri[2], j = tri[0];
                    auto it = sedges.find(vec2i{j, i});
                    if (sedges.find(vec2i{i, j}) == sedges.end() && it == sedges.end())
                        sedges.insert(vec2i{i, j});
                    else if (it != sedges.end()) {
                        auto neighborTriNo = edge2tri[vec2i{j, i}];
                        auto neighborTri = tris[neighborTriNo];

                        auto selfVertNo = tri[1];
                        int neighborVertNo = -1;
                        for (int k = 0; k != 3; ++k)
                            if (neighborTri[k] != i && neighborTri[k] != j) {
                                neighborVertNo = neighborTri[k];
                                break;
                            }
                        bedges.push_back(vec4i{selfVertNo, i, j, neighborVertNo});
                    }
                }
            }
            auto &surfEdges = (*zstris)[ZenoParticles::s_surfEdgeTag];
            surfEdges = typename ZenoParticles::particles_t({{"inds", 2}}, sedges.size(), zs::memsrc_e::host);
            int no = 0;
            auto sv = proxy<execspace_e::host>({}, surfEdges);
            for (auto &&edge : sedges) {
                sv("inds", 0, no, int_c) = edge[0];
                sv("inds", 1, no, int_c) = edge[1];
                no++;
            }
            surfEdges = surfEdges.clone({zs::memsrc_e::device});

            if (withBending) {
                float bendingStrength = 0.f;
                if (has_input<DictObject>("params")) {
                    auto params = get_input<DictObject>("params");
                    auto ps = params->getLiterial<zeno::NumericValue>();
                    if (auto it = ps.find("bending_stiffness"); it != ps.end())
                        bendingStrength = std::get<float>(it->second);
                }

                auto &bendingEdges = (*zstris)[ZenoParticles::s_bendingEdgeTag];
                bendingEdges = typename ZenoParticles::particles_t(
                    {{"inds", 4}, {"k", 1}, {"ra", 1}, {"e", 1}, {"h", 1}}, bedges.size(), zs::memsrc_e::host);

                ompExec(zs::range(bedges.size()), [E, nu, pars = proxy<space>({}, pars), &bedges, &zsmodel,
                                                   bes = proxy<space>({}, bendingEdges),
                                                   bendingStrength](int beNo) mutable {
                    auto bedge = bedges[beNo];
                    bes("inds", 0, beNo, int_c) = bedge[0];
                    bes("inds", 1, beNo, int_c) = bedge[1];
                    bes("inds", 2, beNo, int_c) = bedge[2];
                    bes("inds", 3, beNo, int_c) = bedge[3];
                    /**
                          *             x2 --- x3
                          *            /  \    /
                          *           /    \  /
                          *          x0 --- x1
                          */
                    auto x0 = pars.pack(dim_c<3>, "x", bedge[0]).template cast<double>();
                    auto x1 = pars.pack(dim_c<3>, "x", bedge[1]).template cast<double>();
                    auto x2 = pars.pack(dim_c<3>, "x", bedge[2]).template cast<double>();
                    auto x3 = pars.pack(dim_c<3>, "x", bedge[3]).template cast<double>();

                    auto testGrad = dihedral_angle_gradient(x0, x1, x2, x3);
                    bes("ra", beNo) = (float)zs::dihedral_angle(x0, x1, x2, x3);
                    auto n1 = (x1 - x0).cross(x2 - x0);
                    auto n2 = (x2 - x3).cross(x1 - x3);
                    double e = (x2 - x1).norm();
                    bes("e", beNo) = e;
                    auto h = (n1.norm() + n2.norm()) / (e * 6);
                    bes("h", beNo) = h;
                    if (zs::isnan(testGrad.dot(testGrad)))
                        bes("k", beNo) = 0;
                    else {
                        double k_bend = bendingStrength == 0.f ? ((double)E / (24 * (1 - (double)nu * nu)) *
                                                                  (double)zsmodel->dx * zsmodel->dx * zsmodel->dx)
                                                               : bendingStrength;
                        bes("k", beNo) = k_bend;
                    }
                });
                bendingEdges = bendingEdges.clone({zs::memsrc_e::device});
            }
#endif
            // surface vert indices
            auto &surfVerts = (*zstris)[ZenoParticles::s_surfVertTag];
            surfVerts = typename ZenoParticles::particles_t({{"inds", 1}}, pos.size(), zs::memsrc_e::host);
            ompExec(zs::range(pos.size()), [&, surfVerts = proxy<space>({}, surfVerts)](int pointNo) mutable {
                surfVerts("inds", pointNo) = zs::reinterpret_bits<float>(pointNo);
            });

            pars = pars.clone({zs::memsrc_e::device});
            eles = eles.clone({zs::memsrc_e::device});
            surfEdges = surfEdges.clone({zs::memsrc_e::device});
            surfVerts = surfVerts.clone({zs::memsrc_e::device});
        })(tag);

        set_output("ZSParticles", std::move(zstris));
    }
};

ZENDEFNODE(ToZSSurfaceMesh, {{{"ZSModel"},
                              {"surf (tri) mesh", "prim"},
                              {"float", "rest_shape_scaling", "1.0"},
                              {"bool", "high_precision", "true"},
                              {"bool", "with_bending", "false"},
                              {"DictObject", "params"}},
                             {{"trimesh on gpu", "ZSParticles"}},
                             {},
                             {"FEM"}});

struct MakeSample1dLine : INode {
    void apply() override {
        auto n = get_input2<int>("n");
        auto scale = get_input2<float>("scale");
        vec3f p{0, 0, 0};
        auto seg = scale / n;
        auto prim = std::make_shared<PrimitiveObject>();
        auto &verts = prim->attr<vec3f>("pos");
        auto &lines = prim->lines.values;
        int no = 0;
        verts.push_back(p);
        for (int i = 0; i != n; ++i) {
            p[1] += seg;
            verts.push_back(p);
            lines.push_back(vec2i{i, i + 1});
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(MakeSample1dLine, {{{"int", "n", "1"}, {"float", "scale", "1"}}, {{"line", "prim"}}, {}, {"FEM"}});

struct ToZSStrands : INode {
    using T = float;
    using dtiles_t = typename ZenoParticles::dtiles_t;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;

    void apply() override {
        using namespace zs;
        auto zsmodel = get_input<ZenoConstitutiveModel>("ZSModel");
        auto prim = get_input<PrimitiveObject>("prim");
        const auto &pos = prim->attr<zeno::vec3f>("pos");
        const auto &lines = prim->lines;

        auto ompExec = zs::omp_exec();
        const auto numVerts = pos.size();
        const auto numLines = lines.size();

        auto zsstrands = std::make_shared<ZenoParticles>();
        zsstrands->prim = prim;
        zsstrands->getModel() = *zsmodel;
        zsstrands->category = ZenoParticles::curve;
        zsstrands->sprayedOffset = pos.size();

        std::vector<zs::PropertyTag> tags{
            {"m", 1},       {"x", 3},       {"x0", 3},      {"v", 3}, {"BCbasis", 9} /* normals for slip boundary*/,
            {"BCorder", 1}, {"BCfixed", 1}, {"BCtarget", 3}};
        std::vector<zs::PropertyTag> eleTags{{"vol", 1}, {"k", 1}, {"rl", 1}, {"inds", 2}};

        constexpr auto space = zs::execspace_e::openmp;
        auto &pars = zsstrands->getParticles<true>();
        pars = dtiles_t{tags, pos.size(), zs::memsrc_e::host};
        ompExec(Collapse{pars.size()}, [pars = proxy<space>({}, pars), &pos, &prim](int vi) mutable {
            using vec3 = zs::vec<double, 3>;
            using mat3 = zs::vec<float, 3, 3>;
            using vec3f = zs::vec<float, 3>;
            auto p = vec3f::from_array(pos[vi]);
            pars.template tuple<3>("x", vi) = p;
            pars.template tuple<3>("x0", vi) = p;
            pars.template tuple<3>("v", vi) = vec3::zeros();
            if (prim->has_attr("vel"))
                pars.template tuple<3>("v", vi) = vec3f::from_array(prim->attr<zeno::vec3f>("vel")[vi]);
            // default boundary handling setup
            pars.template tuple<9>("BCbasis", vi) = mat3::identity();
            pars("BCorder", vi) = 0;
            pars("BCfixed", vi) = 0;
            pars.template tuple<3>("BCtarget", vi) = vec3::zeros();
            // computed later
            pars("m", vi) = 0;
        });

        T mu{};
        match([&](auto &elasticModel) { mu = elasticModel.mu; })(zsmodel->getElasticModel());
        zsstrands->elements = typename ZenoParticles::particles_t(eleTags, lines.size(), zs::memsrc_e::host);
        auto &eles = zsstrands->getQuadraturePoints();
        ompExec(Collapse{lines.size()},
                [&zsmodel, pars = proxy<space>({}, pars), eles = proxy<space>({}, eles), &lines, mu](int ei) mutable {
                    for (size_t i = 0; i < 2; ++i)
                        eles("inds", i, ei) = zs::reinterpret_bits<float>(lines[ei][i]);
                    using vec3 = zs::vec<double, 3>;
                    using mat2 = zs::vec<float, 2, 2>;
                    using vec4 = zs::vec<float, 4>;
                    auto line = lines[ei];
                    vec3 xs[2];
                    for (int d = 0; d != 2; ++d) {
                        eles("inds", d, ei) = zs::reinterpret_bits<float>(line[d]);
                        xs[d] = pars.template pack<3>("x", line[d]);
                    }

                    auto rl = (xs[1] - xs[0]).norm();
                    eles("rl", ei) = rl;
                    eles("k", ei) = mu;

                    auto vol = rl * zsmodel->dx * zsmodel->dx;
                    eles("vol", ei) = vol;

                    // vert masses
                    auto vmass = vol * zsmodel->density / 2;
                    for (int d = 0; d != 2; ++d)
                        atomic_add(zs::exec_omp, &pars("m", line[d]), vmass);
#if 0
      if (ei < 10)
        fmt::print("{}-th string rest length: {}, vol: {}, inds: <{}, {}>\n",
                   ei, rl, vol, line[0], line[1]);
#endif
                });

        // surface vert indices
        auto &surfVerts = (*zsstrands)[ZenoParticles::s_surfVertTag];
        surfVerts = typename ZenoParticles::particles_t({{"inds", 1}}, pos.size(), zs::memsrc_e::host);
        ompExec(zs::range(pos.size()), [&, surfVerts = proxy<space>({}, surfVerts)](int pointNo) mutable {
            surfVerts("inds", pointNo) = zs::reinterpret_bits<float>(pointNo);
        });

        pars = pars.clone({zs::memsrc_e::device});
        eles = eles.clone({zs::memsrc_e::device});
        surfVerts = surfVerts.clone({zs::memsrc_e::device});

        set_output("ZSParticles", std::move(zsstrands));
    }
};

ZENDEFNODE(ToZSStrands, {{{"ZSModel"}, {"strand", "prim"}}, {{"strand on gpu", "ZSParticles"}}, {}, {"FEM"}});

} // namespace zeno
