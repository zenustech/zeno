#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include <mshio/mshio.h>

namespace zeno {

struct ReadTetPrim : INode {
  void apply() override {
    auto path = get_input<StringObject>("path")->get();
    auto prim = std::make_shared<PrimitiveObject>();
    auto &pos = prim->attr<vec3f>("pos");
    auto &quads = prim->quads;
    auto ompExec = zs::omp_exec();

    mshio::MshSpec spec = mshio::load_msh(path);
    const auto numVerts = spec.nodes.num_nodes;
    prim->resize(spec.nodes.num_nodes);
    const auto numEles = spec.elements.num_elements;
    quads.resize(numEles);

    /// load vert positions
    ompExec(spec.nodes.entity_blocks, [&](const auto &nodeBlock) {
      const std::size_t stride = 3 + nodeBlock.parametric;
      for (auto tag : nodeBlock.tags) {
        std::size_t j = tag - 1; // [1, N]
        auto x =
            vec3f{nodeBlock.data[j * stride], nodeBlock.data[j * stride + 1],
                  nodeBlock.data[j * stride + 2]};
        pos[j] = x;
      }
    });
    ompExec(spec.elements.entity_blocks, [&](const auto &eleBlock) {
      const std::size_t n = mshio::nodes_per_element(eleBlock.element_type);
      const std::size_t stride = n + 1;
      if (n != 4)
        fmt::print("damn, it\'s not a tet.\n");
      if (eleBlock.element_type != 4)
        fmt::print("not the expected tet.\n");
      for (std::size_t i = 0; i != eleBlock.num_elements_in_block; ++i) {
        std::size_t j = eleBlock.data[i * stride] - 1;
        for (int k = 0; k != n; ++k)
          quads[j][k] = eleBlock.data[j * stride + k + 1];
      }
    });

    if (get_param<bool>("surface_extraction")) {
      using namespace zs;
      zs::HashTable<int, 3, int> surfTable{0};
      constexpr auto space = zs::execspace_e::openmp;
      auto &tris = prim->tris;     // surfaces
      auto &lines = prim->lines;   // surface edges
      auto &points = prim->points; // surface points

      surfTable.resize(ompExec, 4 * numEles);
      surfTable.reset(ompExec, true);
      // compute getsurface
      std::vector<int> tri2tet(4 * numEles);
      ompExec(range(numEles), [table = proxy<space>(surfTable), &quads,
                               &tri2tet](int ei) mutable {
        using table_t = RM_CVREF_T(table);
        using vec3i = zs::vec<int, 3>;
        auto record = [&table, &tri2tet, ei](const vec3i &triInds) mutable {
          if (auto sno = table.insert(triInds); sno != table_t::sentinel_v)
            tri2tet[sno] = ei;
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
#if 1
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
#else
      ompExec(range(numEles), [&](int ei) {
        auto inds = quads[ei];
        tris[ei * 4 + 0] = vec3i{inds[0], inds[2], inds[1]};
        tris[ei * 4 + 1] = vec3i{inds[0], inds[3], inds[2]};
        tris[ei * 4 + 2] = vec3i{inds[0], inds[1], inds[3]};
        tris[ei * 4 + 3] = vec3i{inds[0], inds[2], inds[3]};
      });
      auto scnt = numEles * 4;
      tris.resize(scnt);
#endif
      fmt::print("{} surfaces\n", scnt);

      // surface points
      HashTable<int, 1, int> vertTable{pos.size()};
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
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ReadTetPrim, {/* inputs: */ {
                             {"readpath", "path"},
                         },
                         /* outputs: */
                         {
                             {"primitive", "prim"},
                         },
                         /* params: */
                         {
                             {"bool", "surface_extraction", "1"},
                         },
                         /* category: */
                         {
                             "primitive",
                         }});

} // namespace zeno