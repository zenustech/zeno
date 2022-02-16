#include "../Structures.hpp"
#include "../Utils.hpp"

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
    out->dx = dx;

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

    auto outParticles = std::make_shared<ZenoParticles>();

    // primitive binding
    outParticles->prim = inParticles;
    // model
    outParticles->getModel() = *model;

    /// category, size
    std::size_t size{obj.size()};
    // (mesh）
    std::size_t eleSize{0};
    std::vector<float> dofVol{};
    std::vector<float> eleVol{};
    std::vector<vec3f> elePos{};
    std::vector<vec3f> eleVel{};
    std::vector<std::array<vec3f, 3>> eleD{};

    ZenoParticles::category_e category{ZenoParticles::mpm};
    bool bindMesh = get_input2<int>("category") != ZenoParticles::mpm;
    if (bindMesh) {
      if (quads.size()) {
        category = ZenoParticles::tet;
        eleSize = quads.size();
      } else if (tris.size()) {
        category = ZenoParticles::surface;
        eleSize = tris.size();
      } else if (lines.size()) {
        category = ZenoParticles::curve;
        eleSize = lines.size();
      } else
        throw std::runtime_error("unable to deduce primitive manifold type.");

      dofVol.resize(size, 0.f);

      eleVol.resize(eleSize);
      elePos.resize(eleSize);
      eleVel.resize(eleSize);
      eleD.resize(eleSize);
    }
    outParticles->category = category;

    // per vertex (node) vol, pos, vel
    using namespace zs;
    auto ompExec = zs::omp_exec();

    if (bindMesh) {
      switch (category) {
      // tet
      case ZenoParticles::tet: {
        const auto tetVol = [&obj](vec4i quad) {
          const auto &p0 = obj[quad[0]];
          auto s = cross(obj[quad[2]] - p0, obj[quad[1]] - p0);
          return std::abs(dot(s, obj[quad[3]] - p0)) / 6;
        };
        for (std::size_t i = 0; i != eleSize; ++i) {
          auto quad = quads[i];
          auto v = tetVol(quad);

          eleVol[i] = v;
          elePos[i] =
              (obj[quad[0]] + obj[quad[1]] + obj[quad[2]] + obj[quad[3]]) / 4;
          if (velsPtr)
            eleVel[i] = (velsPtr[quad[0]] + velsPtr[quad[1]] +
                         velsPtr[quad[2]] + velsPtr[quad[3]]) /
                        4;
          eleD[i][0] = obj[quad[1]] - obj[quad[0]];
          eleD[i][1] = obj[quad[2]] - obj[quad[0]];
          eleD[i][2] = obj[quad[3]] - obj[quad[0]];
          for (auto pi : quad)
            dofVol[pi] += v / 4;
        }
      } break;
      // surface
      case ZenoParticles::surface: {
        const auto triArea = [&obj](vec3i tri) {
          using TV3 = zs::vec<float, 3>;
          TV3 p0 = TV3{obj[tri[0]][0], obj[tri[0]][1], obj[tri[0]][2]};
          TV3 p1 = TV3{obj[tri[1]][0], obj[tri[1]][1], obj[tri[1]][2]};
          TV3 p2 = TV3{obj[tri[2]][0], obj[tri[2]][1], obj[tri[2]][2]};
          return (p1 - p0).cross(p2 - p0).norm() * 0.5f;
          // const auto &p0 = obj[tri[0]];
          // return length(cross(obj[tri[1]] - p0, obj[tri[2]] - p0)) * 0.5;
        };
        for (std::size_t i = 0; i != eleSize; ++i) {
          auto tri = tris[i];
          auto v = triArea(tri) * model->dx;
#if 0
          if (i <= 3) {
            for (auto pi : tri)
              fmt::print("vi[{}]: {}, {}, {}\n", pi, obj[pi][0], obj[pi][1],
                         obj[pi][2]);
            fmt::print("tri area: {}, volume: {}, dx: {}\n", triArea(tri), v,
                       model->dx);
            getchar();
          }
#endif
          eleVol[i] = v;
          elePos[i] = (obj[tri[0]] + obj[tri[1]] + obj[tri[2]]) / 3;
          if (velsPtr)
            eleVel[i] =
                (velsPtr[tri[0]] + velsPtr[tri[1]] + velsPtr[tri[2]]) / 3;
          eleD[i][0] = obj[tri[1]] - obj[tri[0]];
          eleD[i][1] = obj[tri[2]] - obj[tri[0]];
          eleD[i][2] = normalize(cross(eleD[i][0], eleD[i][1]));
          for (auto pi : tri)
            dofVol[pi] += v / 3;
        }
      } break;
      // curve
      case ZenoParticles::curve: {
        const auto lineLength = [&obj](vec2i line) {
          return length(obj[line[1]] - obj[line[0]]);
        };
        for (std::size_t i = 0; i != eleSize; ++i) {
          auto line = lines[i];
          auto v = lineLength(line) * model->dx * model->dx;
          eleVol[i] = v;
          elePos[i] = (obj[line[0]] + obj[line[1]]) / 2;
          if (velsPtr)
            eleVel[i] = (velsPtr[line[0]] + velsPtr[line[1]]) / 2;
          eleD[i][0] = obj[line[1]] - obj[line[0]];
          if (auto n = cross(vec3f{0, 1, 0}, eleD[i][0]);
              lengthSquared(n) > zs::limits<float>::epsilon() * 128) {
            eleD[i][1] = normalize(n);
          } else
            eleD[i][1] = normalize(cross(vec3f{1, 0, 0}, eleD[i][0]));
          eleD[i][2] = normalize(cross(eleD[i][0], eleD[i][1]));
          for (auto pi : line)
            dofVol[pi] += v / 2;
        }
      } break;
      default:;
      } // end switch
    }   // end bindmesh

    // particles
    auto &pars = outParticles->getParticles(); // tilevector

    // attributes
    std::vector<zs::PropertyTag> tags{{"mass", 1}, {"pos", 3}, {"vel", 3},
                                      {"vol", 1},  {"C", 9},   {"vms", 1}};
    std::vector<zs::PropertyTag> eleTags{
        {"mass", 1}, {"pos", 3},  {"vel", 3},
        {"vol", 1},  {"C", 9},    {"F", 9},
        {"d", 9},    {"Dinv", 9}, {"inds", (int)category + 1}};

    const bool hasLogJp = model->hasLogJp();
    const bool hasOrientation = model->hasOrientation();
    const bool hasF = model->hasF();

    if (hasF)
      tags.emplace_back(zs::PropertyTag{"F", 9});
    else {
      tags.emplace_back(zs::PropertyTag{"J", 1});
      if (category != ZenoParticles::mpm)
        throw std::runtime_error(
            "mesh particles should not use the 'J' attribute.");
    }

    if (hasOrientation) {
      tags.emplace_back(zs::PropertyTag{"a", 3});
      if (category != ZenoParticles::mpm)
        //
        ;
    }

    if (hasLogJp) {
      tags.emplace_back(zs::PropertyTag{"logJp", 1});
      if (category != ZenoParticles::mpm)
        //
        ;
    }

    // prim attrib tags
    std::vector<zs::PropertyTag> auxAttribs{};
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
    tags.insert(std::end(tags), std::begin(auxAttribs), std::end(auxAttribs));

    fmt::print(
        "{} elements in process. pending {} particles with these attributes.\n",
        eleSize, size);
    for (auto tag : tags)
      fmt::print("tag: [{}, {}]\n", tag.name, tag.numChannels);

    {
      pars = typename ZenoParticles::particles_t{tags, size, memsrc_e::host};
      ompExec(zs::range(size), [pars = proxy<execspace_e::host>({}, pars),
                                hasLogJp, hasOrientation, hasF, &model, &obj,
                                velsPtr, nrmsPtr, &dofVol, category,
                                &inParticles, &auxAttribs](size_t pi) mutable {
        using vec3 = zs::vec<float, 3>;
        using mat3 = zs::vec<float, 3, 3>;

        // volume, mass
        float vol = category == ZenoParticles::mpm ? model->volume : dofVol[pi];
        pars("vol", pi) = vol;
        pars("mass", pi) = vol * model->density;

        // pos
        pars.tuple<3>("pos", pi) = obj[pi];

        // vel
        if (velsPtr != nullptr)
          pars.tuple<3>("vel", pi) = velsPtr[pi];
        else
          pars.tuple<3>("vel", pi) = vec3::zeros();

        // deformation
        if (hasF)
          pars.tuple<9>("F", pi) = mat3::identity();
        else
          pars("J", pi) = 1.;

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
        for (auto &prop : auxAttribs) {
          if (prop.numChannels == 3)
            pars.tuple<3>(prop.name, pi) =
                inParticles->attr<vec3f>(std::string{prop.name})[pi];
          else
            pars(prop.name, pi) =
                inParticles->attr<float>(std::string{prop.name})[pi];
        }
      });

      pars = pars.clone({memsrc_e::um, 0});
    }
    if (bindMesh) {
      outParticles->elements =
          typename ZenoParticles::particles_t{eleTags, eleSize, memsrc_e::host};
      auto &eles = outParticles->getQuadraturePoints(); // tilevector
      ompExec(zs::range(eleSize),
              [eles = proxy<execspace_e::host>({}, eles), &model, velsPtr,
               nrmsPtr, &eleVol, &elePos, &eleVel, &eleD, category, &quads,
               &tris, &lines](size_t ei) mutable {
                using vec3 = zs::vec<float, 3>;
                using mat3 = zs::vec<float, 3, 3>;
                // vol, mass
                eles("vol", ei) = eleVol[ei];
                eles("mass", ei) = eleVol[ei] * model->density;

                // pos
                eles.tuple<3>("pos", ei) = elePos[ei];

                // vel
                if (velsPtr != nullptr)
                  eles.tuple<3>("vel", ei) = eleVel[ei];
                else
                  eles.tuple<3>("vel", ei) = vec3::zeros();

                // deformation
                const auto &D = eleD[ei]; // [col]
                auto Dmat = mat3{D[0][0], D[1][0], D[2][0], D[0][1], D[1][1],
                                 D[2][1], D[0][2], D[1][2], D[2][2]};
                // could qr decomp here first (tech doc)
                eles.tuple<9>("d", ei) = Dmat;
                eles.tuple<9>("Dinv", ei) = zs::inverse(Dmat);
                eles.tuple<9>("F", ei) = mat3::identity();

                // apic transfer
                eles.tuple<9>("C", ei) = mat3::zeros();

                // plasticity

                // element-vertex indices
                if (category == ZenoParticles::tet) {
                  const auto &quad = quads[ei];
                  for (int i = 0; i != 4; ++i) {
                    eles("inds", i, ei) = quad[i];
                  }
                } else if (category == ZenoParticles::surface) {
                  const auto &tri = tris[ei];
                  for (int i = 0; i != 3; ++i) {
                    eles("inds", i, ei) = tri[i];
                  }
                } else if (category == ZenoParticles::curve) {
                  const auto &line = lines[ei];
                  for (int i = 0; i != 2; ++i) {
                    eles("inds", i, ei) = line[i];
                  }
                }
              });
      eles = eles.clone({memsrc_e::um, 0});
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

      // const auto category = parObjPtr->category;
      auto &pos = parObjPtr->prim->attr<vec3f>("pos");
      auto size = pos.size(); // in case zsparticle-mesh is refined
      vec3f *velsPtr{nullptr};
      if (parObjPtr->prim->has_attr("vel"))
        velsPtr = parObjPtr->prim->attr<vec3f>("vel").data();

      // currently only write back pos and vel (if has)
      ompExec(range(size),
              [&, pars = proxy<execspace_e::host>({}, pars)](std::size_t pi) {
                pos[pi] = pars.array<3>("pos", pi);
                if (velsPtr != nullptr)
                  velsPtr[pi] = pars.array<3>("vel", pi);
              });
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
    auto partition = std::make_shared<ZenoPartition>();
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

    auto grid = std::make_shared<ZenoGrid>();
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

struct MakeZSLevelSet : INode {
  void apply() override {
    auto dx = get_input2<float>("dx");

    std::vector<zs::PropertyTag> tags{{"sdf", 1}};

    auto ls = std::make_shared<ZenoLevelSet>();
    ls->transferScheme = get_input2<std::string>("transfer");
    auto cateStr = get_input2<std::string>("category");

    // default is "cellcentered"
    if (cateStr == "staggered")
      tags.emplace_back(zs::PropertyTag{"vel", 3});
    // default is "flip"
    if (ls->transferScheme == "flip")
      tags.emplace_back(zs::PropertyTag{"vdiff", 3});
    else if (ls->transferScheme == "apic")
      ;
    else
      throw std::runtime_error(fmt::format(
          "unrecognized transfer scheme [{}]\n", ls->transferScheme));

    if (cateStr == "collocated")
      ls->getLevelSet() =
          typename ZenoLevelSet::template spls_t<zs::grid_e::collocated>{
              tags, dx, 1, zs::memsrc_e::um, 0};
    else if (cateStr == "cellcentered")
      ls->getLevelSet() =
          typename ZenoLevelSet::template spls_t<zs::grid_e::cellcentered>{
              tags, dx, 1, zs::memsrc_e::um, 0};
    else if (cateStr == "staggered")
      ls->getLevelSet() =
          typename ZenoLevelSet::template spls_t<zs::grid_e::staggered>{
              tags, dx, 1, zs::memsrc_e::um, 0};
    else
      throw std::runtime_error(
          fmt::format("unknown levelset (grid) category [{}].", cateStr));

    zs::match([](const auto &lsPtr) {
      if constexpr (zs::is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>) {
        using spls_t = typename RM_CVREF_T(lsPtr)::element_type;
        fmt::print(
            "levelset [{}] of dx [{}, {}], side_length [{}], block_size [{}]\n",
            spls_t::category, lsPtr->_i2wShat(0, 0), lsPtr->_grid.dx,
            spls_t::side_length, spls_t::block_size);
      } else {
        throw std::runtime_error(
            fmt::format("invalid levelset [{}] initialized in basicls.",
                        zs::get_var_type_str(lsPtr)));
      }
    })(ls->getBasicLevelSet()._ls);
    set_output("ZSLevelSet", std::move(ls));
  }
};
ZENDEFNODE(MakeZSLevelSet, {
                               {{"float", "dx", "0.1"},
                                {"string", "transfer", "flip"},
                                {"string", "category", "cellcentered"}},
                               {"ZSLevelSet"},
                               {},
                               {"SOP"},
                           });

struct ToZSBoundary : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZSBoundary\n");
    auto boundary = std::make_shared<ZenoBoundary>();

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

    boundary->zsls = get_input<ZenoLevelSet>("ZSLevelSet");

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
                                        yprAngles[2], zs::degree_c, zs::ypr_c};
      boundary->R = rot;
    }
    { boundary->omega = zs::AngularVelocity<float, 3>{}; }

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
    auto dt = get_input2<float>("dt");

    // auto oldB = boundary->b;

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
                               {"ZSBoundary", {"float", "dt", "0"}},
                               {"ZSBoundary"},
                               {},
                               {"MPM"},
                           });

/// conversion

struct ZSParticlesToPrimitiveObject : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing "
                                      "ZSParticlesToPrimitiveObject\n");
    auto &zspars = get_input<ZenoParticles>("ZSParticles")->getParticles();
    const auto size = zspars.size();

    auto prim = std::make_shared<PrimitiveObject>();
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