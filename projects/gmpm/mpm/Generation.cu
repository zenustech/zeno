#include "Structures.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ConfigConstitutiveModel : INode {
  void apply() override {
    auto out = std::make_shared<ZenoConstitutiveModel>();

    float dx = get_param<float>("dx");
    if (has_input<NumericObject>("dx"))
      dx = get_input<NumericObject>("dx")->get<float>();

    float ppc = get_param<float>("ppc");
    if (has_input<NumericObject>("ppc"))
      ppc = get_input<NumericObject>("ppc")->get<float>();
    // volume
    out->volume = dx * dx * dx / ppc;

    float density = get_param<float>("density");
    if (has_input<NumericObject>("density"))
      density = get_input<NumericObject>("density")->get<float>();
    // density
    out->density = density;

    float E = get_param<float>("E");
    if (has_input<NumericObject>("E"))
      E = get_input<NumericObject>("E")->get<float>();

    float nu = get_param<float>("nu");
    if (has_input<NumericObject>("nu"))
      nu = get_input<NumericObject>("nu")->get<float>();

    auto typeStr = get_param<std::string>("type");
    // elastic model
    auto &model = out->getElasticModel();
    if (typeStr == "fcr")
      model = zs::FixedCorotated<float>{E, nu};
    else if (typeStr == "nhk")
      model = zs::NeoHookean<float>{E, nu};
    else if (typeStr == "stvk")
      model = zs::StvkWithHencky<float>{E, nu};

    // plastic model
    out->getPlasticModel() = std::monostate{};

    set_output("ZSModel", out);
  }
};

ZENDEFNODE(ConfigConstitutiveModel, {
                                        {"dx", "ppc", "density", "E", "nu"},
                                        {"ZSModel"},
                                        {{"float", "dx", "0.1"},
                                         {"float", "ppc", "8"},
                                         {"float", "density", "1000"},
                                         {"string", "type", "fcr"},
                                         {"float", "E", "10000"},
                                         {"float", "nu", "0.4"}},
                                        {"MPM"},
                                    });

struct ToZSParticles : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZensimParticles\n");
    auto model = get_input<ZenoConstitutiveModel>("ZSModel");

    auto inParticles = get_input<PrimitiveObject>("prim");

    auto &obj = inParticles->attr<vec3f>("pos");
    auto &vels = inParticles->attr<vec3f>("vel");

    const auto size = obj.size();

    auto outParticles = IObject::make<ZenoParticles>();
    // model
    outParticles->getModel() = *model;

    // particles
    auto &pars = outParticles->getParticles(); // tilevector

    std::vector<zs::PropertyTag> tags{
        {"mass", 1}, {"pos", 3}, {"vel", 3}, {"C", 9}, {"vms", 1}};

    const bool hasPlasticity = model->hasPlasticity();
    const bool hasF = model->hasF();

    if (hasF)
      tags.emplace_back(zs::PropertyTag{"F", 9});
    else
      tags.emplace_back(zs::PropertyTag{"J", 1});

    if (hasPlasticity)
      tags.emplace_back(zs::PropertyTag{"logJp", 1});

    fmt::print("{} particles of these tags\n", size);
    for (auto tag : tags)
      fmt::print("tag: [{}, {}]\n", tag.name, tag.numChannels);

    {
      using namespace zs;
      pars = typename ZenoParticles::particles_t{tags, size, memsrc_e::host};

      auto ompExec = zs::omp_exec();
      ompExec(zs::range(size),
              [pars = proxy<execspace_e::host>({}, pars), hasPlasticity, hasF,
               &inParticles, &model, &obj, &vels](size_t pi) mutable {
                using vec3 = zs::vec<float, 3>;
                using mat3 = zs::vec<float, 3, 3>;
                pars("mass", pi) = model->volume * model->density;
                pars.tuple<3>("pos", pi) = obj[pi];
                pars.tuple<3>("vel", pi) = vels[pi];
                pars.tuple<9>("C", pi) = mat3::zeros();
                if (hasF)
                  pars.tuple<9>("F", pi) = mat3::identity();
                else
                  pars("J", pi) = 1.;

                if (hasPlasticity)
                  pars("logJp", pi) = 0;
                pars("vms", pi) = 0; // vms
              });

      pars = pars.clone({memsrc_e::um, 0});
    }

    fmt::print(fg(fmt::color::cyan), "done executing ToZensimParticles\n");
    set_output("ZSParticles", outParticles);
  }
};

ZENDEFNODE(ToZSParticles, {
                              {"ZSModel", "prim"},
                              {"ZSParticles"},
                              {},
                              {"MPM"},
                          });

struct MakeZSPartition : INode {
  void apply() override {
    auto partition = IObject::make<ZenoPartition>();
    partition->get() =
        typename ZenoPartition::table_t{(std::size_t)1, zs::memsrc_e::um, 0};
    set_output("ZSPartition", partition);
  }
};
ZENDEFNODE(MakeZSPartition, {
                                {},
                                {"ZSPartition"},
                                {},
                                {"MPM"},
                            });

struct MakeZSGrid : INode {
  void apply() override {
    auto dx = get_param<float>("dx");
    if (has_input("dx"))
      dx = get_input<NumericObject>("dx")->get<float>();

    auto grid = IObject::make<ZenoGrid>();
    grid->get() = typename ZenoGrid::grid_t{
        {{"m", 1}, {"v", 3}}, dx, 1, zs::memsrc_e::um, 0};

    using traits = zs::grid_traits<typename ZenoGrid::grid_t>;
    fmt::print("grid of dx [{}], side_length [{}], block_size [{}]\n",
               grid->get().dx, traits::side_length, traits::block_size);
    set_output("ZSGrid", grid);
  }
};
ZENDEFNODE(MakeZSGrid, {
                           {"dx"},
                           {"ZSGrid"},
                           {{"float", "dx", "0.1"}},
                           {"MPM"},
                       });

struct ToZSBoundary : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZSBoundary\n");
    auto boundary = zeno::IObject::make<ZenoBoundary>();

    auto type = get_param<std::string>("type");
    auto queryType = [&type]() -> zs::collider_e {
      if (type == "sticky" || type == "Sticky")
        return zs::collider_e::Sticky;
      else if (type == "slip" || type == "Slip")
        return zs::collider_e::Slip;
      else if (type == "separate" || type == "Separate")
        return zs::collider_e::Separate;
      return zs::collider_e::Sticky;
    };
    // pass in FloatGrid::Ptr
    auto &ls = get_input<ZenoLevelSet>("ZSLevelSet")->getLevelSet();

    boundary->levelset = &ls;
    boundary->type = queryType();
    // *boundary = ZenoBoundary{&ls, queryType()};
    fmt::print(fg(fmt::color::cyan), "done executing ToZSBoundary\n");
    set_output("ZSBoundary", boundary);
  }
};
ZENDEFNODE(ToZSBoundary, {
                             {"ZSLevelSet"},
                             {"ZSBoundary"},
                             {{"string", "type", "sticky"}},
                             {"MPM"},
                         });

/// conversion

struct ZSParticlesToPrimitiveObject : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing "
                                      "ZSParticlesToPrimitiveObject\n");
    auto &zspars = get_input<ZenoParticles>("ZSParticles")->getParticles();
    const auto size = zspars.size();

    auto prim = zeno::IObject::make<PrimitiveObject>();
    prim->resize(size);

    using namespace zs;
    auto cudaExec = cuda_exec().device(0);

    static_assert(sizeof(zs::vec<float, 3>) == sizeof(zeno::vec3f),
                  "zeno::vec3f != zs::vec<float, 3>");
    for (auto &&prop : zspars.getPropertyTags()) {
      if (prop.numChannels == 3) {
        zs::Vector<zs::vec<float, 3>> dst{size, memsrc_e::device, 0};
        cudaExec(zs::range(size),
                 [zspars = zs::proxy<execspace_e::cuda>({}, zspars),
                  dst = zs::proxy<execspace_e::cuda>(dst),
                  name = prop.name] __device__(size_t pi) mutable {
                   dst[pi] = zspars.pack<3>(name, pi);
                 });
        copy(zs::mem_device,
             prim->add_attr<zeno::vec3f>(prop.name.asString()).data(),
             dst.data(), sizeof(zeno::vec3f) * size);
      } else if (prop.numChannels == 1) {
        zs::Vector<float> dst{size, memsrc_e::device, 0};
        cudaExec(zs::range(size),
                 [zspars = zs::proxy<execspace_e::cuda>({}, zspars),
                  dst = zs::proxy<execspace_e::cuda>(dst),
                  name = prop.name] __device__(size_t pi) mutable {
                   dst[pi] = zspars(name, pi);
                 });
        copy(zs::mem_device, prim->add_attr<float>(prop.name.asString()).data(),
             dst.data(), sizeof(float) * size);
      }
    }
    fmt::print(fg(fmt::color::cyan), "done executing "
                                     "ZSParticlesToPrimitiveObject\n");
    set_output("prim", prim);
  }
};

ZENDEFNODE(ZSParticlesToPrimitiveObject, {
                                             {"ZSParticles"},
                                             {"prim"},
                                             {},
                                             {"MPM"},
                                         });

struct WriteZSParticles : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing WriteZSParticles\n");
    auto &pars = get_input<ZenoParticles>("ZSParticles")->getParticles();
    auto path = get_param<std::string>("path");
    auto cudaExec = zs::cuda_exec().device(0);
    zs::Vector<zs::vec<float, 3>> pos{pars.size(), zs::memsrc_e::um, 0};
    zs::Vector<float> vms{pars.size(), zs::memsrc_e::um, 0};
    cudaExec(zs::range(pars.size()),
             [pos = zs::proxy<zs::execspace_e::cuda>(pos),
              vms = zs::proxy<zs::execspace_e::cuda>(vms),
              pars = zs::proxy<zs::execspace_e::cuda>(
                  {}, pars)] __device__(size_t pi) mutable {
               pos[pi] = pars.pack<3>("pos", pi);
               vms[pi] = pars("vmstress", pi);
             });
    std::vector<std::array<float, 3>> posOut(pars.size());
    std::vector<float> vmsOut(pars.size());
    copy(zs::mem_device, posOut.data(), pos.data(),
         sizeof(zeno::vec3f) * pars.size());
    copy(zs::mem_device, vmsOut.data(), vms.data(),
         sizeof(float) * pars.size());

    zs::write_partio_with_stress<float, 3>(path, posOut, vmsOut);
    fmt::print(fg(fmt::color::cyan), "done executing WriteZSParticles\n");
  }
};

ZENDEFNODE(WriteZSParticles, {
                                 {"ZSParticles"},
                                 {},
                                 {{"string", "path", ""}},
                                 {"MPM"},
                             });

} // namespace zeno