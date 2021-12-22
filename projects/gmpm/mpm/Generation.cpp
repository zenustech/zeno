#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/VdbSampler.h"
#include <zeno/VDBGrid.h>

namespace zeno {

struct ToZSLevelSet : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZSLevelSet\n");
    auto ls = IObject::make<ZenoLevelSet>();

    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;

    if (has_input<VDBFloatGrid>("VDBGrid")) {
      // pass in FloatGrid::Ptr
      zs::OpenVDBStruct gridPtr = get_input<VDBFloatGrid>("VDBGrid")->m_grid;
      ls->getLevelSet() = basic_ls_t{zs::convert_floatgrid_to_sparse_levelset(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0})};
    } else if (has_input<VDBFloat3Grid>("VDBGrid")) {
      // pass in FloatGrid::Ptr
      zs::OpenVDBStruct gridPtr = get_input<VDBFloat3Grid>("VDBGrid")->m_grid;
      ls->getLevelSet() = basic_ls_t{zs::convert_vec3fgrid_to_sparse_levelset(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0})};
    } else {
      auto path = get_param<std::string>("path");
#if 0
      auto gridPtr = zs::loadFloatGridFromVdbFile(path);
      ls->getLevelSet() = zs::convert_floatgrid_to_sparse_levelset(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0});
#else
      auto gridPtr = zs::load_vec3fgrid_from_vdb_file(path);
      ls->getLevelSet() = basic_ls_t{zs::convert_vec3fgrid_to_sparse_levelset(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0})};
#endif
    }

    fmt::print(fg(fmt::color::cyan), "done executing ToZSLevelSet\n");
    set_output("ZSLevelSet", std::move(ls));
  }
};
ZENDEFNODE(ToZSLevelSet, {
                             {"VDBGrid"},
                             {"ZSLevelSet"},
                             {{"string", "path", ""}},
                             {"MPM"},
                         });

struct ComposeSdfVelField : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ComposeSdfVelField\n");
    auto ls = IObject::make<ZenoLevelSet>();

    std::shared_ptr<ZenoLevelSet> sdfLsPtr{};
    std::shared_ptr<ZenoLevelSet> velLsPtr{};

    using sdf_vel_ls_t = typename ZenoLevelSet::sdf_vel_ls_t;

    if (has_input<ZenoLevelSet>("ZSSdfField")) {
      sdfLsPtr = get_input<ZenoLevelSet>("ZSSdfField");
    }
    if (has_input<ZenoLevelSet>("ZSVelField")) {
      velLsPtr = get_input<ZenoLevelSet>("ZSVelField");
    }
    if (velLsPtr) {
      if (!sdfLsPtr->holdsBasicLevelSet() || !velLsPtr->holdsBasicLevelSet()) {
        auto msg = fmt::format("sdfField is {}a basic levelset {} and velField "
                               "is {}a basic levelset.\n",
                               sdfLsPtr->holdsBasicLevelSet() ? "" : "not ",
                               velLsPtr->holdsBasicLevelSet() ? "" : "not ");
        throw std::runtime_error(msg);
      }
      ls->getLevelSet() = sdf_vel_ls_t{sdfLsPtr->getBasicLevelSet(),
                                       velLsPtr->getBasicLevelSet()};
    } else {
      if (!sdfLsPtr->holdsBasicLevelSet()) {
        auto msg = fmt::format("sdfField is {}a basic levelset.\n",
                               sdfLsPtr->holdsBasicLevelSet() ? "" : "not ");
        throw std::runtime_error(msg);
      }
      ls->getLevelSet() = sdf_vel_ls_t{sdfLsPtr->getBasicLevelSet()};
    }

    fmt::print(fg(fmt::color::cyan), "done executing ComposeSdfVelField\n");
    set_output("ZSLevelSet", std::move(ls));
  }
};
ZENDEFNODE(ComposeSdfVelField, {
                                   {"ZSSdfField", "ZSVelField"},
                                   {"ZSLevelSet"},
                                   {},
                                   {"MPM"},
                               });

struct ZSLevelSetToVDBGrid : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSLevelSetToVDBGrid\n");
    auto vdb = IObject::make<VDBFloatGrid>();

    if (has_input<ZenoLevelSet>("ZSLevelSet")) {
      auto ls = get_input<ZenoLevelSet>("ZSLevelSet");
      if (ls->holdsSparseLevelSet()) {
        vdb->m_grid =
            zs::convert_sparse_levelset_to_floatgrid(ls->getSparseLevelSet())
                .as<openvdb::FloatGrid::Ptr>();
      } else
        ZS_WARN("The current input levelset is not a sparse levelset!");
    }

    fmt::print(fg(fmt::color::cyan), "done executing ZSLevelSetToVDBGrid\n");
    set_output("VDBFloatGrid", std::move(vdb));
  }
};
ZENDEFNODE(ZSLevelSetToVDBGrid, {
                                    {"ZSLevelSet"},
                                    {"VDBFloatGrid"},
                                    {},
                                    {"MPM"},
                                });

} // namespace zeno