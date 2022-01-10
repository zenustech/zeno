#include "../Utils.hpp"
#include "Structures.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ConfigConstitutiveModel : INode {
  void apply() override {
    auto out = std::make_shared<ZenoConstitutiveModel>();

    float dx = get_input2<float>("dx");

    // volume
    out->volume = dx * dx * dx / get_input2<float>("ppc");

    // density
    out->density = get_input2<float>("density");

    // constitutive models
    auto params = has_input("params") ? get_input<DictObject>("params")
                                      : std::make_shared<DictObject>();
    float E = get_input2<float>("E");

    float nu = get_input2<float>("nu");

    auto typeStr = get_input2<std::string>("type");
    // elastic model
    auto &model = out->getElasticModel();

    if (typeStr == "fcr")
      model = zs::FixedCorotated<float>{E, nu};
    else if (typeStr == "nhk")
      model = zs::NeoHookean<float>{E, nu};
    else if (typeStr == "stvk")
      model = zs::StvkWithHencky<float>{E, nu};
    else
      throw std::runtime_error(fmt::format(
          "unrecognized (isotropic) elastic model [{}]\n", typeStr));

    // aniso elastic model
    const auto get_arg = [&params](const char *const tag, auto type) {
      using T = typename RM_CVREF_T(type)::type;
      std::optional<T> ret{};
      if (auto it = params->lut.find(tag); it != params->lut.end())
        ret = safe_any_cast<T>(it->second);
      return ret;
    };
    auto anisoTypeStr = get_input2<std::string>("aniso");
    if (anisoTypeStr == "arap") { // a (fiber direction)
      float strength = get_arg("strength", zs::wrapt<float>{}).value_or(10.f);
      out->getAnisoElasticModel() = zs::AnisotropicArap<float>{E, nu, strength};
    } else
      out->getAnisoElasticModel() = std::monostate{};

    // plastic model
    auto plasticTypeStr = get_input2<std::string>("plasticity");
    if (plasticTypeStr == "nadp") {
      model = zs::StvkWithHencky<float>{E, nu};
      float fa = get_arg("friction_angle", zs::wrapt<float>{}).value_or(35.f);
      out->getPlasticModel() = zs::NonAssociativeDruckerPrager<float>{fa};
    } else if (plasticTypeStr == "navm") {
      model = zs::StvkWithHencky<float>{E, nu};
      float ys = get_arg("yield_stress", zs::wrapt<float>{}).value_or(1e5f);
      out->getPlasticModel() = zs::NonAssociativeVonMises<float>{ys};
    } else if (plasticTypeStr == "nacc") { // logjp
      model = zs::StvkWithHencky<float>{E, nu};
      float fa = get_arg("friction_angle", zs::wrapt<float>{}).value_or(35.f);
      float beta = get_arg("beta", zs::wrapt<float>{}).value_or(2.f);
      float xi = get_arg("xi", zs::wrapt<float>{}).value_or(1.f);
      out->getPlasticModel() =
          zs::NonAssociativeCamClay<float>{fa, beta, xi, 3, true};
    } else
      out->getPlasticModel() = std::monostate{};

    set_output("ZSModel", out);
  }
};

ZENDEFNODE(ConfigConstitutiveModel,
           {
               {{"float", "dx", "0.1"},
                {"float", "ppc", "8"},
                {"float", "density", "1000"},
                {"string", "type", "fcr"},
                {"string", "aniso", "none"},
                {"string", "plasticity", "none"},
                {"float", "E", "10000"},
                {"float", "nu", "0.4"},
                {"DictObject:NumericObject", "params"}},
               {"ZSModel"},
               {},
               {"MPM"},
           });

struct ToZSParticles : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZensimParticles\n");
    auto model = get_input<ZenoConstitutiveModel>("ZSModel");

    // primitive
    auto inParticles = get_input<PrimitiveObject>("prim");
    auto &obj = inParticles->attr<vec3f>("pos");
    vec3f *velsPtr{nullptr};
    if (inParticles->has_attr("vel"))
      velsPtr = inParticles->attr<vec3f>("vel").data();
    vec3f *nrmsPtr{nullptr};
    if (inParticles->has_attr("nrm"))
      nrmsPtr = inParticles->attr<vec3f>("nrm").data();
    auto &quads = inParticles->quads;
    auto &tris = inParticles->tris;
    auto &lines = inParticles->lines;

    auto outParticles = IObject::make<ZenoParticles>();

    // primitive binding
    outParticles->prim = inParticles;
    // model
    outParticles->getModel() = *model;
    // category
    outParticles->category =
        static_cast<ZenoParticles::category_e>(get_input2<int>("category"));

    std::size_t size = obj.size();
    // per vertex (node) vol, pos, vel
    std::vector<float> eleVol(size, 0);

    std::vector<vec3f> elePos{};
    std::vector<vec3f> eleVel{};

    using namespace zs;
    auto ompExec = zs::omp_exec();

    if (auto category = outParticles->category;
        category != ZenoParticles::mpm) {
      // tet
      if (quads.size()) {
        if (category == ZenoParticles::element) {
          size = quads.size();
          elePos.resize(size);
          eleVel.resize(size);
        }
        const auto tetVol = [&obj](vec4i quad) {
          const auto &p0 = obj[quad[0]];
          auto s = cross(obj[quad[2]] - p0, obj[quad[1]] - p0);
          return std::abs(dot(s, obj[quad[3]] - p0)) / 6;
        };
        for (std::size_t i = 0; i != quads.size(); ++i) {
          auto quad = quads[i];
          auto v = tetVol(quad);
          if (category == ZenoParticles::vertex) {
            for (auto pi : quad)
              eleVol[pi] += v / 4;
          } else if (category == ZenoParticles::element) {
            eleVol[i] = v;
            elePos[i] =
                (obj[quad[0]] + obj[quad[1]] + obj[quad[2]] + obj[quad[3]]) / 4;
            if (velsPtr)
              eleVel[i] = (velsPtr[quad[0]] + velsPtr[quad[1]] +
                           velsPtr[quad[2]] + velsPtr[quad[3]]) /
                          4;
          }
        }
      }
      // surface
      else if (tris.size()) {
        if (category == ZenoParticles::element) {
          size = tris.size();
          elePos.resize(size);
          eleVel.resize(size);
        }
        const auto triArea = [&obj](vec3i tri) {
          const auto &p0 = obj[tri[0]];
          return length(cross(obj[tri[1]] - p0, obj[tri[2]] - p0)) * 0.5;
        };
        for (std::size_t i = 0; i != tris.size(); ++i) {
          auto tri = tris[i];
          auto area = triArea(tri);
          if (category == ZenoParticles::vertex) {
            for (auto pi : tri)
              eleVol[pi] += area / 3;
          } else if (category == ZenoParticles::element) {
            eleVol[i] = area;
            elePos[i] = (obj[tri[0]] + obj[tri[1]] + obj[tri[2]]) / 3;
            if (velsPtr)
              eleVel[i] =
                  (velsPtr[tri[0]] + velsPtr[tri[1]] + velsPtr[tri[2]]) / 3;
          }
        }
      }
      // strand
      else if (lines.size()) {
        if (category == ZenoParticles::element) {
          size = lines.size();
          elePos.resize(size);
          eleVel.resize(size);
        }
        const auto lineLength = [&obj](vec2i line) {
          return length(obj[line[1]] - obj[line[0]]);
        };
        for (std::size_t i = 0; i != lines.size(); ++i) {
          auto line = lines[i];
          auto len = lineLength(line);
          if (category == ZenoParticles::vertex) {
            for (auto pi : line)
              eleVol[pi] += len / 2;
          } else if (category == ZenoParticles::vertex) {
            eleVol[i] = len;
            elePos[i] = (obj[line[0]] + obj[line[1]]) / 2;
            if (velsPtr)
              eleVel[i] = (velsPtr[line[0]] + velsPtr[line[1]]) / 2;
          }
        }
      }
    }

    // particles
    auto &pars = outParticles->getParticles(); // tilevector

    std::vector<zs::PropertyTag> tags{{"mass", 1}, {"pos", 3}, {"vel", 3},
                                      {"vol", 1},  {"C", 9},   {"vms", 1}};

    const bool hasLogJp = model->hasLogJp();
    const bool hasOrientation =
        model->hasOrientation() ||
        outParticles->category == ZenoParticles::element;
    const bool hasF = model->hasF();
    const bool hasDeformation =
        outParticles->category == ZenoParticles::mpm ||
        outParticles->category == ZenoParticles::element;

    if (hasDeformation) {
      if (hasF)
        tags.emplace_back(zs::PropertyTag{"F", 9});
      else
        tags.emplace_back(zs::PropertyTag{"J", 1});
    }

    if (hasOrientation)
      tags.emplace_back(zs::PropertyTag{"a", 3});

    if (hasLogJp)
      tags.emplace_back(zs::PropertyTag{"logJp", 1});

    // tag assembly
    std::vector<zs::PropertyTag> auxAttribs{};
    if (outParticles->category == ZenoParticles::mpm ||
        outParticles->category == ZenoParticles::vertex) {
      for (auto &&[key, arr] : inParticles->verts.attrs) {
        const auto checkDuplication = [&tags](const std::string &name) {
          for (std::size_t i = 0; i != tags.size(); ++i)
            if (tags[i].name == name.data())
              return true;
          return false;
        };
        if (checkDuplication(key))
          continue;
        const auto &k{key};
        match(
            [&k, &auxAttribs](const std::vector<vec3f> &vals) {
              auxAttribs.push_back(PropertyTag{k, 3});
            },
            [&k, &auxAttribs](const std::vector<float> &vals) {
              auxAttribs.push_back(PropertyTag{k, 1});
            },
            [&k, &auxAttribs](const std::vector<vec3i> &vals) {},
            [&k, &auxAttribs](const std::vector<int> &vals) {},
            [](...) {
              throw std::runtime_error(
                  "what the heck is this type of attribute!");
            })(arr);
      }
    }
    tags.insert(std::end(tags), std::begin(auxAttribs), std::end(auxAttribs));

    fmt::print("pending {} particles with these attributes\n", size);
    for (auto tag : tags)
      fmt::print("tag: [{}, {}]\n", tag.name, tag.numChannels);

    {
      pars = typename ZenoParticles::particles_t{tags, size, memsrc_e::host};

      ompExec(zs::range(size),
              [pars = proxy<execspace_e::host>({}, pars), hasLogJp,
               hasOrientation, hasF, hasDeformation, &model, &obj, velsPtr,
               nrmsPtr, &eleVol, &elePos, &eleVel,
               category = (int)outParticles->category, &inParticles,
               &auxAttribs](size_t pi) mutable {
                using vec3 = zs::vec<float, 3>;
                using mat3 = zs::vec<float, 3, 3>;
                float vol{};
                // vol, pos, vel
                if (category == ZenoParticles::mpm) {
                  vol = model->volume;
                  pars.tuple<3>("pos", pi) = obj[pi];
                  if (velsPtr != nullptr)
                    pars.tuple<3>("vel", pi) = velsPtr[pi];
                  else
                    pars.tuple<3>("vel", pi) = vec3::zeros();
                } else if (category == ZenoParticles::vertex) {
                  // iter
                  vol = eleVol[pi]; // different
                  pars.tuple<3>("pos", pi) = obj[pi];
                  if (velsPtr != nullptr)
                    pars.tuple<3>("vel", pi) = velsPtr[pi];
                  else
                    pars.tuple<3>("vel", pi) = vec3::zeros();
                } else if (category == ZenoParticles::element) {
                  vol = eleVol[pi];
                  pars.tuple<3>("pos", pi) = elePos[pi];
                  if (velsPtr != nullptr)
                    pars.tuple<3>("vel", pi) = eleVel[pi];
                  else
                    pars.tuple<3>("vel", pi) = vec3::zeros();
                }

                pars("vol", pi) = vol;
                pars("mass", pi) = vol * model->density;

                // deformation
                if (hasDeformation) {
                  if (hasF)
                    pars.tuple<9>("F", pi) = mat3::identity();
                  else
                    pars("J", pi) = 1.;
                }

                // apic transfer
                pars.tuple<9>("C", pi) = mat3::zeros();

                // orientation
                if (hasOrientation) {
                  if (nrmsPtr != nullptr) {
                    const auto n_ = nrmsPtr[pi];
                    const auto n = vec3{n_[0], n_[1], n_[2]};
                    constexpr auto up = vec3{0, 1, 0};
                    if (!parallel(n, up)) {
                      auto side = cross(up, n);
                      auto a = cross(side, n);
                      pars.tuple<3>("a", pi) = a;
                    } else
                      pars.tuple<3>("a", pi) = vec3{0, 0, 1};
                  } else
                    pars.tuple<3>("a", pi) = vec3::zeros();
                }

                // plasticity
                if (hasLogJp)
                  pars("logJp", pi) = -0.04;
                pars("vms", pi) = 0; // vms

                // additional attributes
                if (category == ZenoParticles::mpm ||
                    category == ZenoParticles::vertex) {
                  for (auto &prop : auxAttribs) {
                    if (prop.numChannels == 3)
                      pars.tuple<3>(prop.name, pi) =
                          inParticles->attr<vec3f>(std::string{prop.name})[pi];
                    else
                      pars(prop.name, pi) =
                          inParticles->attr<float>(std::string{prop.name})[pi];
                  }
                }
              });

      pars = pars.clone({memsrc_e::um, 0});
    }

    fmt::print(fg(fmt::color::cyan), "done executing ToZensimParticles\n");
    set_output("ZSParticles", outParticles);
  }
};

ZENDEFNODE(ToZSParticles, {
                              {"ZSModel", "prim", {"int", "category", "0"}},
                              {"ZSParticles"},
                              {},
                              {"MPM"},
                          });

/// this requires further polishing
struct UpdatePrimitiveFromZSParticles : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing UpdatePrimitiveFromZSParticles\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");

    using namespace zs;
    auto ompExec = zs::omp_exec();

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      if (parObjPtr->prim.get() == nullptr)
        continue;

      const auto category = parObjPtr->category;
      auto &pos = parObjPtr->prim->attr<vec3f>("pos");
      vec3f *velsPtr{nullptr};
      if (parObjPtr->prim->has_attr("vel"))
        velsPtr = parObjPtr->prim->attr<vec3f>("vel").data();

      auto size = pars.size();

      if (category == ZenoParticles::mpm || category == ZenoParticles::vertex) {
        // currently only write back pos and vel (if has)
        ompExec(range(size),
                [&, pars = proxy<execspace_e::host>({}, pars)](std::size_t pi) {
                  pos[pi] = pars.array<3>("pos", pi);
                  if (velsPtr != nullptr)
                    velsPtr[pi] = pars.array<3>("vel", pi);
                });
      } else if (category == ZenoParticles::element) {
        ;
      }
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing UpdatePrimitiveFromZSParticles\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(UpdatePrimitiveFromZSParticles, {
                                               {"ZSParticles"},
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
    auto dx = get_input2<float>("dx");

    std::vector<zs::PropertyTag> tags{{"m", 1}, {"v", 3}};

    auto grid = IObject::make<ZenoGrid>();
    grid->transferScheme = get_input2<std::string>("transfer");
    // default is "apic"
    if (grid->transferScheme == "flip")
      tags.emplace_back(zs::PropertyTag{"vdiff", 3});
    else if (grid->transferScheme == "apic")
      ;
    else
      throw std::runtime_error(fmt::format(
          "unrecognized transfer scheme [{}]\n", grid->transferScheme));

    grid->get() = typename ZenoGrid::grid_t{tags, dx, 1, zs::memsrc_e::um, 0};

    using traits = zs::grid_traits<typename ZenoGrid::grid_t>;
    fmt::print("grid of dx [{}], side_length [{}], block_size [{}]\n",
               grid->get().dx, traits::side_length, traits::block_size);
    set_output("ZSGrid", grid);
  }
};
ZENDEFNODE(MakeZSGrid,
           {
               {{"float", "dx", "0.1"}, {"string", "transfer", "apic"}},
               {"ZSGrid"},
               {},
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

    // translation
    if (has_input("translation")) {
      auto b = get_input<NumericObject>("translation")->get<vec3f>();
      boundary->b = zs::vec<float, 3>{b[0], b[1], b[2]};
    }
    if (has_input("translation_rate")) {
      auto dbdt = get_input<NumericObject>("translation_rate")->get<vec3f>();
      boundary->dbdt = zs::vec<float, 3>{dbdt[0], dbdt[1], dbdt[2]};
      // fmt::print("dbdt assigned as {}, {}, {}\n", boundary->dbdt[0],
      //            boundary->dbdt[1], boundary->dbdt[2]);
    }
    // scale
    if (has_input("scale")) {
      auto s = get_input<NumericObject>("scale")->get<float>();
      boundary->s = s;
    }
    if (has_input("scale_rate")) {
      auto dsdt = get_input<NumericObject>("scale_rate")->get<float>();
      boundary->dsdt = dsdt;
    }
    // rotation
    if (has_input("ypr_angles")) {
      auto yprAngles = get_input<NumericObject>("ypr_angles")->get<vec3f>();
      auto rot = zs::Rotation<float, 3>{yprAngles[0], yprAngles[1],
                                        yprAngles[2], zs::degree_v, zs::ypr_v};
      boundary->R = rot;
    }
    { boundary->omega = zs::AngularVelocity<float, 3>{}; }

    // *boundary = ZenoBoundary{&ls, queryType()};
    fmt::print(fg(fmt::color::cyan), "done executing ToZSBoundary\n");
    set_output("ZSBoundary", boundary);
  }
};
ZENDEFNODE(ToZSBoundary, {
                             {"ZSLevelSet", "translation", "translation_rate",
                              "scale", "scale_rate", "ypr_angles"},
                             {"ZSBoundary"},
                             {{"string", "type", "sticky"}},
                             {"MPM"},
                         });

struct StepZSBoundary : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing StepZSBoundary\n");

    auto boundary = get_input<ZenoBoundary>("ZSBoundary");
    auto dt = get_param<float>("dt");
    if (has_input("dt"))
      dt = get_input<NumericObject>("dt")->get<float>();

    auto oldB = boundary->b;

    boundary->s += boundary->dsdt * dt;
    boundary->b += boundary->dbdt * dt;

#if 0
    auto b = boundary->b;
    auto dbdt = boundary->dbdt;
    auto delta = dbdt * dt;
    fmt::print("({}, {}, {}) + ({}, {}, {}) * {} -> ({}, {}, {})\n", oldB[0],
               oldB[1], oldB[2], dbdt[0], dbdt[1], dbdt[2], dt, delta[0],
               delta[1], delta[2]);
#endif

    fmt::print(fg(fmt::color::cyan), "done executing StepZSBoundary\n");
    set_output("ZSBoundary", boundary);
  }
};
ZENDEFNODE(StepZSBoundary, {
                               {"ZSBoundary", "dt"},
                               {"ZSBoundary"},
                               {{"float", "dt", "0"}},
                               {"MPM"},
                           });

/// conversion

struct ZSParticlesToPrimitiveObject : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing "
                                      "ZSParticlesToPrimitiveObject\n");
    auto &zspars = get_input<ZenoParticles>("ZSParticles")->getParticles();
    const auto size = zspars.size();

    auto prim = IObject::make<PrimitiveObject>();
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
               vms[pi] = pars("vms", pi);
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

struct ComputeVonMises : INode {
  template <typename Model>
  void computeVms(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                  typename ZenoParticles::particles_t &pars, int option) {
    using namespace zs;
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 model, option] __device__(size_t pi) mutable {
      auto F = pars.pack<3, 3>("F", pi);
      auto [U, S, V] = math::svd(F);
      auto cauchy = model.dpsi_dsigma(S) * S / S.prod();

      auto diff = cauchy;
      for (int d = 0; d != 3; ++d)
        diff(d) -= cauchy((d + 1) % 3);

      auto vms = ::sqrt(diff.l2NormSqr() * 0.5f);
      pars("vms", pi) = option ? ::log10(vms + 1) : vms;
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ComputeVonMises\n");
    auto zspars = get_input<ZenoParticles>("ZSParticles");
    auto &pars = zspars->getParticles();
    auto model = zspars->getModel();
    auto option = get_param<int>("by_log1p(base10)");

    auto cudaExec = zs::cuda_exec().device(0);
    zs::match([&](auto &elasticModel) {
      computeVms(cudaExec, elasticModel, pars, option);
    })(model.getElasticModel());

    set_output("ZSParticles", std::move(zspars));
    fmt::print(fg(fmt::color::cyan), "done executing ComputeVonMises\n");
  }
};

ZENDEFNODE(ComputeVonMises, {
                                {"ZSParticles"},
                                {"ZSParticles"},
                                {{"int", "by_log1p(base10)", "1"}},
                                {"MPM"},
                            });

} // namespace zeno