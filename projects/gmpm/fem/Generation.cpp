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
    std::vector<float> pointAreas;
    std::vector<vec2i> lines;
    std::vector<float> lineAreas;
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
      pointAreas.resize(svcnt, 0.f);
      copy(mem_host, points.data(), vertTable._activeKeys.data(),
           sizeof(int) * svcnt);
      fmt::print("{} surface verts\n", svcnt);

      // surface edges
      Vector<int> surfEdgeCnt{1};
      surfEdgeCnt.setVal(0);
      auto dupEdgeCnt = edgeTable.size();
      std::vector<int> dupEdgeToSurfEdge(dupEdgeCnt, -1);
      lines.resize(dupEdgeCnt);
      ompExec(range(dupEdgeCnt), [edgeTable = proxy<space>(edgeTable), &lines,
                                  surfEdgeCnt = surfEdgeCnt.data(),
                                  &dupEdgeToSurfEdge](int edgeNo) mutable {
        using vec2i = zs::vec<int, 2>;
        vec2i edge = edgeTable._activeKeys[edgeNo];
        using table_t = RM_CVREF_T(edgeTable);
        if (auto eno = edgeTable.query(vec2i{edge[1], edge[0]});
            eno == table_t::sentinel_v || // opposite edge not exists
            (eno != table_t::sentinel_v &&
             edge[0] < edge[1])) { // opposite edge does exist
          auto no = atomic_add(exec_omp, surfEdgeCnt, 1);
          lines[no] = zeno::vec2i{edge[0], edge[1]};
          dupEdgeToSurfEdge[edgeNo] = no;
        }
      });
      auto secnt = surfEdgeCnt.getVal();
      lines.resize(secnt);
      lineAreas.resize(secnt, 0.f);
      fmt::print("{} surface edges\n", secnt);

      ompExec(tris,
              [&, vertTable = proxy<space>(vertTable),
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
    std::vector<zs::PropertyTag> tags{{"x", 3}, {"v", 3}};
    std::vector<zs::PropertyTag> eleTags{{"inds", (int)3}};

    // verts
    zsbou->particles = std::make_shared<typename ZenoParticles::particles_t>(
        tags, numVerts, memsrc_e::host);
    auto &pars = zsbou->getParticles(); // tilevector
    ompExec(zs::range(numVerts), [pars = proxy<execspace_e::openmp>({}, pars),
                                  &pos, velsPtr](int pi) mutable {
      using vec3 = zs::vec<float, 3>;
      // pos
      pars.tuple<3>("x", pi) = pos[pi];
      // vel
      if (velsPtr != nullptr)
        pars.tuple<3>("v", pi) = velsPtr[pi];
      else
        pars.tuple<3>("v", pi) = vec3::zeros();
    });

    // elements
    zsbou->elements =
        typename ZenoParticles::particles_t{eleTags, numEles, memsrc_e::host};
    auto &eles = zsbou->getQuadraturePoints(); // tilevector
    ompExec(zs::range(numEles), [pars = proxy<execspace_e::openmp>({}, pars),
                                 eles = proxy<execspace_e::openmp>({}, eles),
                                 &tris](size_t ei) mutable {
      // element-vertex indices
      // inds
      const auto &tri = tris[ei];
      for (int i = 0; i != 3; ++i)
        eles("inds", i, ei) = reinterpret_bits<float>(tri[i]);
    });

    /// surface edges
    zs::HashTable<int, 2, int> edgeTable{0};
    constexpr auto space = zs::execspace_e::openmp;

    edgeTable.resize(ompExec, numEles);
    edgeTable.reset(ompExec, true);
    ompExec(range(numEles), [table = proxy<space>(edgeTable),
                             &tris](int ei) mutable {
      using table_t = RM_CVREF_T(table);
      using vec2i = zs::vec<int, 2>;
      auto record = [&table, ei](const vec2i &edgeInds) mutable {
        if (auto sno = table.insert(edgeInds); sno != table_t::sentinel_v)
          ;
        else
          printf("ridiculous, more than one surf tri share the same edge!");
      };
      auto inds = tris[ei];
      record(vec2i{inds[0], inds[1]});
      record(vec2i{inds[1], inds[2]});
      record(vec2i{inds[2], inds[0]});
    });
    //
    Vector<int> edgeCnt{1, memsrc_e::host};
    edgeCnt.setVal(0);

    // Vector<zs::vec<int, 2>> surfEdges{(std::size_t)edgeTable.size()};
    auto &surfEdges = (*zsbou)[ZenoParticles::s_surfEdgeTag];
    surfEdges = typename ZenoParticles::particles_t(
        {{"inds", 2}}, edgeTable.size(), zs::memsrc_e::host);

    ompExec(range(edgeTable.size()), [table = proxy<space>(edgeTable),
                                      edgeCnt = edgeCnt.data(),
                                      surfEdges = proxy<space>({}, surfEdges)](
                                         int ei) mutable {
      using vec2i = zs::vec<int, 2>;
      auto edgeInds = table._activeKeys[ei];
      using table_t = RM_CVREF_T(table);
      if (table.query(vec2i{edgeInds[1], edgeInds[0]}) == table_t::sentinel_v) {
        auto no = atomic_add(exec_omp, edgeCnt, 1);
        surfEdges.tuple<2>("inds", no) =
            edgeInds.template reinterpret_bits<float>();
      }
    });
    auto secnt = edgeCnt.getVal();
    surfEdges.resize(secnt);
    fmt::print("{} surface edges\n", secnt);

    eles = eles.clone({memsrc_e::device, 0});
    pars = pars.clone({memsrc_e::device, 0});
    surfEdges = surfEdges.clone({memsrc_e::device, 0});

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

    auto ompExec = zs::omp_exec();
    const auto numVerts = pos.size();
    const auto numEles = quads.size();

    auto zstets = std::make_shared<ZenoParticles>();
    zstets->prim = prim;
    zstets->getModel() = *zsmodel;
    zstets->category = ZenoParticles::tet;
    zstets->sprayedOffset = pos.size();

    std::vector<zs::PropertyTag> tags{
        {"m", 1},
        {"x", 3},
        {"x0", 3},
        {"v", 3},
        {"BCbasis", 9} /* normals for slip boundary*/,
        {"BCorder", 1},
        {"BCtarget", 3}};
    std::vector<zs::PropertyTag> eleTags{{"vol", 1}, {"IB", 9}, {"inds", 4}};

    constexpr auto space = zs::execspace_e::openmp;
    zstets->particles = std::make_shared<typename ZenoParticles::particles_t>(
        tags, pos.size(), zs::memsrc_e::host);
    auto &pars = zstets->getParticles();
    ompExec(zs::range(pos.size()),
            [&, pars = proxy<space>({}, pars)](int vi) mutable {
              using vec3 = zs::vec<float, 3>;
              using mat3 = zs::vec<float, 3, 3>;
              auto p = vec3{pos[vi][0], pos[vi][1], pos[vi][2]};
              pars.tuple<3>("x", vi) = p;
              pars.tuple<3>("x0", vi) = p;
              pars.tuple<3>("v", vi) = vec3::zeros();
              if (prim->has_attr("vel")) {
                auto vel = prim->attr<zeno::vec3f>("vel")[vi];
                pars.tuple<3>("v", vi) = vec3{vel[0], vel[1], vel[2]};
              }
              // default boundary handling setup
              pars.tuple<9>("BCbasis", vi) = mat3::identity();
              pars("BCorder", vi) = reinterpret_bits<float>(0);
              pars.tuple<3>("BCtarget", vi) = vec3::zeros();
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
    auto &surfaces = (*zstets)[ZenoParticles::s_surfTriTag];
    surfaces = typename ZenoParticles::particles_t({{"inds", 3}}, tris.size(),
                                                   zs::memsrc_e::host);
    ompExec(zs::range(tris.size()),
            [&, surfaces = proxy<space>({}, surfaces)](int triNo) mutable {
              auto tri = tris[triNo];
              for (int i = 0; i != 3; ++i)
                surfaces("inds", i, triNo) =
                    zs::reinterpret_bits<float>(tri[i]);
            });
    auto &surfEdges = (*zstets)[ZenoParticles::s_surfEdgeTag];
    surfEdges = typename ZenoParticles::particles_t(
        {{"inds", 2}, {"w", 1}}, lines.size(), zs::memsrc_e::host);
    const auto &lineAreas = lines.attr<float>("area");
    ompExec(zs::range(lines.size()),
            [&, surfEdges = proxy<space>({}, surfEdges)](int lineNo) mutable {
              auto line = lines[lineNo];
              for (int i = 0; i != 2; ++i)
                surfEdges("inds", i, lineNo) =
                    zs::reinterpret_bits<float>(line[i]);
              surfEdges("w", lineNo) = lineAreas[lineNo]; // line area (weight)
            });
    auto &surfVerts = (*zstets)[ZenoParticles::s_surfVertTag];
    surfVerts = typename ZenoParticles::particles_t(
        {{"inds", 1}, {"w", 1}}, points.size(), zs::memsrc_e::host);
    const auto &pointAreas = points.attr<float>("area");
    ompExec(zs::range(points.size()),
            [&, surfVerts = proxy<space>({}, surfVerts)](int pointNo) mutable {
              auto point = points[pointNo];
              surfVerts("inds", pointNo) = zs::reinterpret_bits<float>(point);
              surfVerts("w", pointNo) =
                  pointAreas[pointNo]; // point area (weight)
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

struct ToZSTriMesh : INode {
  using T = float;
  using dtiles_t = zs::TileVector<T,32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<T, 3>;  

  void apply() override {
    using namespace zs;
    // auto zsmodel = get_input<ZenoConstitutiveModel>("ZSModel");
    auto prim = get_input<PrimitiveObject>("prim");
    const auto &pos = prim->attr<zeno::vec3f>("pos");
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

    std::vector<zs::PropertyTag> tags{
      {"x",3} 
    };
    std::vector<zs::PropertyTag> eleTags{{"inds",3}};

    constexpr auto space = zs::execspace_e::openmp;
    zstris->particles = std::make_shared<tiles_t>(tags,pos.size(),zs::memsrc_e::host);
    auto& pars = zstris->getParticles();
    ompExec(Collapse{pars.size()},
      [pars = proxy<space>({},pars), &pos](int vi) mutable {
        pars.tuple<3>("x",vi) = vec3{pos[vi][0],pos[vi][1],pos[vi][2]};
      });

    zstris->elements = typename ZenoParticles::particles_t(eleTags,tris.size(),zs::memsrc_e::host);
    auto& eles = zstris->getQuadraturePoints();
    ompExec(Collapse{tris.size()},
      [eles = proxy<space>({},eles), &tris](int ei) mutable {
        for(size_t i = 0;i < 3;++i)
          eles("inds",i,ei) = zs::reinterpret_bits<float>(tris[ei][i]);
      });

    pars = pars.clone({zs::memsrc_e::device,0});
    eles = eles.clone({zs::memsrc_e::device,0});



    set_output("ZSParticles",std::move(zstris));
  }
};

ZENDEFNODE(ToZSTriMesh, {{{"surf (tri) mesh", "prim"}},
                            {{"trimesh on gpu", "ZSParticles"}},
                            {},
                            {"FEM"}});


} // namespace zeno