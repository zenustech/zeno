#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/VDBGrid.h>

#include "../utils.cuh"
#include "zeno/utils/log.h"

namespace zeno {

struct ZSMakeAdaptiveGrid : INode {
  void apply() override {
    auto attr = get_input2<std::string>("Attribute");
    auto dx = get_input2<float>("Dx");
    auto bg = get_input2<float>("background");
    auto type = get_input2<std::string>("type");
    auto structure = get_input2<std::string>("structure");

    auto zsAG = std::make_shared<ZenoAdaptiveGrid>();

    int nc = 1;
    if (type == "scalar")
      nc = 1;
    else if (type == "vector3")
      nc = 3;

    using namespace zs;
    if (auto str = get_input2<std::string>("category"); str == "vdb") {
      auto &ag = zsAG->beginVdbGrid();

      ag.level(dim_c<0>) = RM_CVREF_T(ag.level(dim_c<0>))(
          {{attr, nc}}, 0, zs::memsrc_e::device, 0);
      ag.level(dim_c<1>) = RM_CVREF_T(ag.level(dim_c<1>))(
          {{attr, nc}}, 0, zs::memsrc_e::device, 0);
      ag.level(dim_c<2>) = RM_CVREF_T(ag.level(dim_c<2>))(
          {{attr, nc}}, 0, zs::memsrc_e::device, 0);
      ag.scale(dx);
      ag._background = bg;

      if (structure == "vertex-centered") {
        auto trans = zs::vec<float, 3>::constant(-dx / 2);
        // zs::vec<float, 3> trans{-dx / 2.f, -dx / 2.f, -dx / 2.f};
        ag.translate(trans);
      }
    } else {
      auto &ag = zsAG->beginTileTree();

      ag.level(dim_c<0>) = RM_CVREF_T(ag.level(dim_c<0>))(
          {{attr, nc}}, 0, zs::memsrc_e::device, 0);
      ag.level(dim_c<1>) = RM_CVREF_T(ag.level(dim_c<1>))(
          {{attr, nc}}, 0, zs::memsrc_e::device, 0);
      ag.level(dim_c<2>) = RM_CVREF_T(ag.level(dim_c<2>))(
          {{attr, nc}}, 0, zs::memsrc_e::device, 0);
      ag.scale(dx);
      ag._background = bg;

      if (structure == "vertex-centered") {
        auto trans = zs::vec<float, 3>::constant(-dx / 2);
        // zs::vec<float, 3> trans{-dx / 2.f, -dx / 2.f, -dx / 2.f};
        ag.translate(trans);
      }
    }

    set_output("Grid", zsAG);
  }
};

ZENDEFNODE(ZSMakeAdaptiveGrid, {/* inputs: */
                                {
                                    {"string", "Attribute", ""},
                                    {"float", "Dx", "1.0"},
                                    {"float", "background", "0"},
                                    {"enum scalar vector3", "type", "scalar"},
                                    {"enum cell-centered vertex-centered",
                                     "structure", "cell-centered"},
                                    {"enum vdb tile_tree", "category", "vdb"},
                                },
                                /* outputs: */
                                {"Grid"},
                                /* params: */
                                {},
                                /* category: */
                                {"Eulerian"}});

struct ZSAdaptiveGridToVDB : INode {
  void apply() override {
    auto zs_grid = get_input<ZenoAdaptiveGrid>("AdaptiveGrid");
    auto attr = get_input2<std::string>("Attribute");
    auto VDBGridClass = get_input2<std::string>("VDBGridClass");
    auto VDBGridName = get_input2<std::string>("VDBGridName");

    if (attr.empty())
      attr = "sdf";

    using namespace zs;
    std::vector<u32> nodeCnts(3);
    auto converter = [&](ZenoAdaptiveGrid::vdb_t &ag) {
      auto attrTag = src_tag(zs_grid, attr);
      auto num_ch = ag.getPropertySize(attrTag);
      if (VDBGridClass == "STAGGERED" && num_ch != 3) {
        throw std::runtime_error(
            "The size of Attribute is not 3 when grid_type is STAGGERED!");
      }
      if constexpr (false) {
        ag.nodeCount(nodeCnts);
        fmt::print(fg(fmt::color::green), "vdb cnts: {}, {}, {}\n", nodeCnts[0],
                   nodeCnts[1], nodeCnts[2]);

        ZenoAdaptiveGrid::att_t att;
        auto ag_ = ag; // ag.clone({memsrc_e::host, -1});
        ag_.restructure(zs::cuda_exec(), att);
        att.restructure(zs::cuda_exec(), ag_);

        ag_.nodeCount(nodeCnts);
        fmt::print(fg(fmt::color::green), "(restred) ag cnts: {}, {}, {}\n",
                   nodeCnts[0], nodeCnts[1], nodeCnts[2]);

        ag = ag_.clone({memsrc_e::device, 0});
      }

      if (num_ch == 3) {
        auto vdb_ =
            zs::convert_adaptive_grid_to_float3grid(ag, attrTag, VDBGridName);
        auto vdb_grid = std::make_shared<VDBFloat3Grid>();
        vdb_grid->m_grid = vdb_.as<openvdb::Vec3fGrid::Ptr>();

        set_output("VDB", vdb_grid);
      } else {
        zs::u32 gridClass = 0;
        if (VDBGridClass == "UNKNOWN")
          gridClass = 0;
        else if (VDBGridClass == "LEVEL_SET")
          gridClass = 1;
        else if (VDBGridClass == "FOG_VOLUME")
          gridClass = 2;

        auto vdb_ = zs::convert_adaptive_grid_to_floatgrid(
            ag, attrTag, gridClass, VDBGridName);

        auto vdb_grid = std::make_shared<VDBFloatGrid>();
        vdb_grid->m_grid = vdb_.as<openvdb::FloatGrid::Ptr>();

        set_output("VDB", vdb_grid);
      }
    };
    match(converter, [&](ZenoAdaptiveGrid::att_t &ag) {
      ZenoAdaptiveGrid::vdb_t vdb;
      if (ag.memspace() == memsrc_e::device) {
        auto pol = cuda_exec();
        ag.restructure(pol, vdb);
        // restructure_adaptive_grid(pol, ag, vdb);
      } else {
        auto pol = cuda_exec();
        auto ag_ = ag.clone({memsrc_e::device, 0});
        ag_.restructure(pol, vdb);
        // restructure_adaptive_grid(pol, ag, vdb);
      }
      converter(vdb);
    })(zs_grid->ag);
  }
};

ZENDEFNODE(ZSAdaptiveGridToVDB,
           {/* inputs: */
            {"AdaptiveGrid",
             {"string", "Attribute", ""},
             {"enum UNKNOWN LEVEL_SET FOG_VOLUME STAGGERED", "VDBGridClass",
              "LEVEL_SET"},
             {"string", "VDBGridName", "AdaptiveGrid"}},
            /* outputs: */
            {"VDB"},
            /* params: */
            {},
            /* category: */
            {"Eulerian"}});

struct ZSVDBToAdaptiveGrid : INode {
  void apply() override {
    auto vdb = get_input<VDBGrid>("VDB");
    auto attr = get_input2<std::string>("Attribute");
    if (attr.empty())
      attr = "sdf";

    if (has_input("AdaptiveGrid")) {
      auto zs_grid = get_input<ZenoAdaptiveGrid>("AdaptiveGrid");
      ZenoAdaptiveGrid::vdb_t &ag = zs_grid->getVdbGrid();

      int num_ch;
      if (vdb->getType() == "FloatGrid")
        num_ch = 1;
      else if (vdb->getType() == "Vec3fGrid")
        num_ch = 3;
      else
        throw std::runtime_error("Input VDB must be a FloatGrid or Vec3fGrid!");

      auto attrTag = src_tag(zs_grid, attr);
      if (ag.hasProperty(attrTag)) {
        if (num_ch != ag.getPropertySize(attrTag)) {
          throw std::runtime_error(
              fmt::format("The channel number of [{}] doesn't match!", attr));
        }
      } else {
        ag.append_channels(zs::cuda_exec(), {{attrTag, num_ch}});
      }

#if 1
      if (num_ch == 1) {
        auto vdb_ = std::dynamic_pointer_cast<VDBFloatGrid>(vdb);
        zs::assign_floatgrid_to_adaptive_grid(vdb_->m_grid, ag, attrTag);
      } else {
        auto vdb_ = std::dynamic_pointer_cast<VDBFloat3Grid>(vdb);
        zs::assign_float3grid_to_adaptive_grid(vdb_->m_grid, ag, attrTag);
      }

      set_output("AdaptiveGrid", zs_grid);
#else
      throw std::runtime_error(
          "openvdb grid assigning to zs adaptive grid not yet implemented");
#endif
    } else {
      ZenoAdaptiveGrid::vdb_t ag;

      auto vdbType = vdb->getType();
      if (vdbType == "FloatGrid") {
        auto vdb_ = std::dynamic_pointer_cast<VDBFloatGrid>(vdb);
        ag = zs::convert_floatgrid_to_adaptive_grid(
            vdb_->m_grid, zs::MemoryHandle{zs::memsrc_e::device, 0}, attr);
      } else if (vdbType == "Vec3fGrid") {
        auto vdb_ = std::dynamic_pointer_cast<VDBFloat3Grid>(vdb);
        ag = zs::convert_float3grid_to_adaptive_grid(
            vdb_->m_grid, zs::MemoryHandle{zs::memsrc_e::device, 0}, attr);
      } else {
        throw std::runtime_error("Input VDB must be a FloatGrid or Vec3fGrid!");
      }

      auto zsSPG = std::make_shared<ZenoAdaptiveGrid>();
      zsSPG->ag = std::move(ag);

      set_output("AdaptiveGrid", zsSPG);
    }
  }
};

ZENDEFNODE(ZSVDBToAdaptiveGrid,
           {/* inputs: */
            {"VDB", "AdaptiveGrid", {"string", "Attribute", ""}},
            /* outputs: */
            {"AdaptiveGrid"},
            /* params: */
            {},
            /* category: */
            {"Eulerian"}});

} // namespace zeno