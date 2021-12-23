#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/VdbSampler.h"
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>

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

struct EnqueueLevelSetSequence : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing EnqueueLevelSetSequence\n");

    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
    using sdf_vel_ls_t = typename ZenoLevelSet::sdf_vel_ls_t;
    using transition_ls_t = typename ZenoLevelSet::transition_ls_t;

    std::shared_ptr<ZenoLevelSet> zsls{};
    if (has_input<ZenoLevelSet>("ZSLevelSetSequence"))
      zsls = get_input<ZenoLevelSet>("ZSLevelSetSequence");
    else {
      zsls = IObject::make<ZenoLevelSet>();
      zsls->levelset = transition_ls_t{};
    }
    auto &lsseq = zsls->getLevelSetSequence();

    if (has_input<ZenoLevelSet>("ZSLevelSet")) {
      auto &ls = get_input<ZenoLevelSet>("ZSLevelSet")->getLevelSet();
      match(
          [&lsseq](basic_ls_t &basicLs) { lsseq.push(sdf_vel_ls_t{basicLs}); },
          [&lsseq](sdf_vel_ls_t &field) { // recommend
            lsseq.push(field);            // also reset alpha in the meantime
          },
          [&lsseq](transition_ls_t &seq) {
            lsseq._fields.insert(lsseq._fields.end(), seq._fields.begin(),
                                 seq._fields.end());
          })(ls);
    }

    if constexpr (false) {
      fmt::print("done enqueueing. {} levelsets in the sequence. ratio: {}, "
                 "stepDt: {}\n",
                 lsseq._fields.size(), lsseq._alpha, lsseq._stepDt);
      getchar();
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing EnqueueLevelSetSequence\n");
    set_output("ZSLevelSetSequence", std::move(zsls));
  }
};
ZENDEFNODE(EnqueueLevelSetSequence, {
                                        {"ZSLevelSetSequence", "ZSLevelSet"},
                                        {"ZSLevelSetSequence"},
                                        {},
                                        {"MPM"},
                                    });

/// update levelsetsequence state
struct UpdateLevelSetSequence : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing UpdateLevelSetSequence\n");

    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
    using sdf_vel_ls_t = typename ZenoLevelSet::sdf_vel_ls_t;
    using transition_ls_t = typename ZenoLevelSet::transition_ls_t;

    std::shared_ptr<ZenoLevelSet> zsls{};
    if (has_input<ZenoLevelSet>("ZSLevelSetSequence"))
      zsls = get_input<ZenoLevelSet>("ZSLevelSetSequence");
    else {
      zsls = IObject::make<ZenoLevelSet>();
      zsls->levelset = transition_ls_t{};
    }
    auto &lsseq = zsls->getLevelSetSequence();

    float stepDt = 0, alpha = 0;
    if constexpr (false) { // debug
      std::string id = "dt";
      fmt::print("raw has_input({}): {}\n", id, has_input(id));
      getchar();
      fmt::print("raw has_input2({}): {}\n", id, has_input2(id));
      getchar();

      auto tmp = get_input2(id);
      {
        auto tt = get_input<NumericObject>(id);
        fmt::print("retrieved value: {}\n", tt->get<float>());
        puts("pass get_input test.");
        getchar();
      }
      using T = std::shared_ptr<IObject>;
      using V = any_underlying_type_t<T>;
      fmt::print("typeid(V) = [{}], tmp = [{}] ref [{}]\n", typeid(V).name(),
                 tmp.type().name(), typeid(tmp).name());
      getchar();

      auto tmp1 = zs_silent_any_cast<T>(tmp);
      fmt::print("direct silen_any_cast: [{}]\n", get_var_type_str(tmp1));
      getchar();

      using U = typename remove_shared_ptr<T>::type;
      fmt::print(
          "get_input2 return [{}], underlying type (T, V, U): [{}, {}, {}]\n",
          get_var_type_str(tmp), get_type_str<T>(), get_type_str<V>(),
          get_type_str<U>());

      decltype(auto) v = std::any_cast<V const &>(tmp);
      auto ptr = std::dynamic_pointer_cast<U>(v);
      fmt::print("silent any cast result: {} ({})\n", (void *)ptr.get(),
                 (bool)ptr);

      fmt::print("raw has_input2<shared_ptr<IObj>>({}): {}\n", id,
                 has_input2<std::shared_ptr<IObject>>(id));
      getchar();

      auto obj = get_input(id);
      auto p = std::dynamic_pointer_cast<NumericObject>(std::move(obj));
      fmt::print("dynamically casted pointer: {} ({})\n", (void *)p.get(),
                 (bool)p);
    }

    if (has_input("dt")) {
      stepDt = get_input<NumericObject>("dt")->get<float>();
      lsseq.setStepDt(stepDt);
    }

    if (has_input("alpha")) {
      alpha = get_input<NumericObject>("alpha")->get<float>();
      lsseq.advance(alpha);
    }

    if constexpr (false) { // debug
      fmt::print(
          "done updating. {} levelsets in the sequence. ratio: {} (+ {}), "
          "stepDt: {} (= {})\n",
          lsseq._fields.size(), lsseq._alpha, alpha, lsseq._stepDt, stepDt);
      getchar();
    }

    fmt::print(fg(fmt::color::cyan), "done executing UpdateLevelSetSequence\n");
    set_output("ZSLevelSetSequence", std::move(zsls));
  }
};
ZENDEFNODE(UpdateLevelSetSequence, {
                                       {"ZSLevelSetSequence", "dt", "alpha"},
                                       {"ZSLevelSetSequence"},
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