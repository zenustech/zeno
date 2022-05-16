#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
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
    std::vector<vec2i> lines;
    std::vector<vec3i> tris;
    {
      using namespace zs;
      zs::HashTable<int, 3, int> surfTable{0};
      constexpr auto space = zs::execspace_e::openmp;

      surfTable.resize(ompExec, 4 * numEles);
      surfTable.reset(ompExec, true);
      // compute getsurface
      // std::vector<int> tri2tet(4 * numEles);
      ompExec(range(numEles), [table = proxy<space>(surfTable),
                               &quads](int ei) mutable {
        using table_t = RM_CVREF_T(table);
        using vec3i = zs::vec<int, 3>;
        auto record = [&table, ei](const vec3i &triInds) mutable {
          if (auto sno = table.insert(triInds); sno != table_t::sentinel_v)
            ; // tri2tet[sno] = ei;
          else
            printf("ridiculous, more than one tet share the same surface!");
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
              [table = proxy<space>(surfTable), surfCnt = surfCnt.data(),
               &tris](int i) mutable {
                using vec3i = zs::vec<int, 3>;
                auto triInds = table._activeKeys[i];
                using table_t = RM_CVREF_T(table);
                if (table.query(vec3i{triInds[2], triInds[1], triInds[0]}) ==
                        table_t::sentinel_v &&
                    table.query(vec3i{triInds[1], triInds[0], triInds[2]}) ==
                        table_t::sentinel_v &&
                    table.query(vec3i{triInds[0], triInds[2], triInds[1]}) ==
                        table_t::sentinel_v)
                  tris[atomic_add(exec_omp, surfCnt, 1)] =
                      zeno::vec3i{triInds[0], triInds[1], triInds[2]};
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
              [vertTable = proxy<space>(vertTable),
               edgeTable = proxy<space>(edgeTable)](vec3i triInds) mutable {
                using vec1i = zs::vec<int, 1>;
                using vec2i = zs::vec<int, 2>;
                for (int d = 0; d != 3; ++d) {
                  vertTable.insert(vec1i{triInds[d]});
                  edgeTable.insert(vec2i{triInds[d], triInds[(d + 1) % 3]});
                }
              });
      auto svcnt = vertTable.size();
      points.resize(svcnt);
      copy(mem_host, points.data(), vertTable._activeKeys.data(),
           sizeof(int) * svcnt);
      fmt::print("{} surface verts\n", svcnt);

      // surface edges
      Vector<int> surfEdgeCnt{1};
      surfEdgeCnt.setVal(0);
      auto dupEdgeCnt = edgeTable.size();
      lines.resize(dupEdgeCnt);
      ompExec(range(dupEdgeCnt),
              [edgeTable = proxy<space>(edgeTable), &lines,
               surfEdgeCnt = surfEdgeCnt.data()](int edgeNo) mutable {
                using vec2i = zs::vec<int, 2>;
                vec2i edge = edgeTable._activeKeys[edgeNo];
                using table_t = RM_CVREF_T(edgeTable);
                if (auto eno = edgeTable.query(vec2i{edge[1], edge[0]});
                    eno == table_t::sentinel_v || // opposite edge not exists
                    (eno != table_t::sentinel_v &&
                     edge[0] < edge[1])) // opposite edge does exist
                  lines[atomic_add(exec_omp, surfEdgeCnt, 1)] =
                      zeno::vec2i{edge[0], edge[1]};
              });
      auto secnt = surfEdgeCnt.getVal();
      lines.resize(secnt);
      fmt::print("{} surface edges\n", secnt);
    }
    if (includeTris)
      prim->tris.values = tris; // surfaces
    if (includeLines)
      prim->lines.values = lines; // surfaces edges
    if (includePoints)
      prim->points.values = points; // surfaces points
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ExtractMeshSurface, {{{"quad (tet) mesh", "prim"}},
                                {{"mesh with surface topos", "prim"}},
                                {{"enum all point edge surface", "op", "all"}},
                                {"primitive"}});

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

    auto ompExec = zs::omp_exec();
    const auto numVerts = pos.size();
    const auto numEles = quads.size();

    auto zstets = std::make_shared<ZenoParticles>();
    zstets->prim = prim;
    zstets->getModel() = *zsmodel;
    zstets->category = ZenoParticles::tet;
    zstets->sprayedOffset = pos.size();

    std::vector<zs::PropertyTag> tags{{"m", 1}, {"x", 3}, {"v", 3}};
    std::vector<zs::PropertyTag> eleTags{{"vol", 1}, {"IB", 9}, {"inds", 4}};

    constexpr auto space = zs::execspace_e::openmp;
    zstets->particles = std::make_shared<typename ZenoParticles::particles_t>(
        tags, pos.size(), zs::memsrc_e::host);
    auto &pars = zstets->getParticles();
    ompExec(zs::range(pos.size()),
            [&, pars = proxy<space>({}, pars)](int vi) mutable {
              using vec3 = zs::vec<float, 3>;
              auto p = vec3{pos[vi][0], pos[vi][1], pos[vi][2]};
              pars.tuple<3>("x", vi) = p;
              if (prim->has_attr("vel")) {
                auto vel = prim->attr<zeno::vec3f>("vel")[vi];
                pars.tuple<3>("v", vi) = vec3{vel[0], vel[1], vel[2]};
              }
              pars.tuple<3>("v", vi) = vec3::zeros();
              // computed later
              pars("m", vi) = 0.f;
            });
    zstets->elements = typename ZenoParticles::particles_t(
        eleTags, quads.size(), zs::memsrc_e::host);
    auto &eles = zstets->getQuadraturePoints();
    ompExec(zs::range(eles.size()),
            [&, pars = proxy<space>({}, pars),
             eles = proxy<space>({}, eles)](int ei) mutable {
              using vec3 = zs::vec<float, 3>;
              using mat3 = zs::vec<float, 3, 3>;
              using vec4 = zs::vec<float, 4>;
              auto quad = quads[ei];
              vec3 xs[4];
              for (int d = 0; d != 4; ++d) {
                eles("inds", d, ei) = zs::reinterpret_bits<float>(quad[d]);
                xs[d] = pars.pack<3>("x", quad[d]);
              }

              vec3 ds[3] = {xs[1] - xs[0], xs[2] - xs[0], xs[3] - xs[0]};
              mat3 D{};
              for (int d = 0; d != 3; ++d)
                for (int i = 0; i != 3; ++i)
                  D(d, i) = ds[i][d];
              eles.tuple<9>("IB", ei) = zs::inverse(D);
              auto vol = zs::abs(zs::determinant(D)) / 6;
              eles("vol", ei) = vol;
              // vert masses
              auto vmass = vol * zsmodel->density / 4;
              for (int d = 0; d != 4; ++d)
                atomic_add(zs::exec_omp, &pars("m", quad[d]), vmass);
            });
    // surface info
    auto &surfaces = (*zstets)["surfaces"];
    surfaces = typename ZenoParticles::particles_t({{"inds", 3}}, tris.size(),
                                                   zs::memsrc_e::host);
    ompExec(zs::range(tris.size()),
            [&, surfaces = proxy<space>({}, surfaces)](int triNo) mutable {
              auto tri = tris[triNo];
              for (int i = 0; i != 3; ++i)
                surfaces("inds", i, triNo) =
                    zs::reinterpret_bits<float>(tri[i]);
            });
    auto &surfEdges = (*zstets)["surfEdges"];
    surfEdges = typename ZenoParticles::particles_t({{"inds", 2}}, lines.size(),
                                                    zs::memsrc_e::host);
    ompExec(zs::range(lines.size()),
            [&, surfEdges = proxy<space>({}, surfEdges)](int lineNo) mutable {
              auto line = lines[lineNo];
              for (int i = 0; i != 2; ++i)
                surfEdges("inds", i, lineNo) =
                    zs::reinterpret_bits<float>(line[i]);
            });
    auto &surfVerts = (*zstets)["surfVerts"];
    surfVerts = typename ZenoParticles::particles_t(
        {{"inds", 1}}, points.size(), zs::memsrc_e::host);
    ompExec(zs::range(points.size()),
            [&, surfVerts = proxy<space>({}, surfVerts)](int pointNo) mutable {
              auto point = points[pointNo];
              surfVerts("inds", pointNo) = zs::reinterpret_bits<float>(point);
            });

    pars = pars.clone({zs::memsrc_e::device, 0});
    eles = eles.clone({zs::memsrc_e::device, 0});
    surfaces = surfaces.clone({zs::memsrc_e::device, 0});
    surfEdges = surfEdges.clone({zs::memsrc_e::device, 0});
    surfVerts = surfVerts.clone({zs::memsrc_e::device, 0});

    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(ToZSTetrahedra, {{{"ZSModel"}, {"quad (tet) mesh", "prim"}},
                            {{"tetmesh on gpu", "ZSParticles"}},
                            {},
                            {"FEM"}});

} // namespace zeno