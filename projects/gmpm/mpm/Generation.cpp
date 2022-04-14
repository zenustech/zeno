#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct PoissonDiskSample : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green), "begin executing PoissonDiskSample\n");
    auto ls = std::make_shared<ZenoLevelSet>();

    zs::OpenVDBStruct gridPtr{};
    if (has_input<VDBFloatGrid>("VDBGrid"))
      gridPtr = get_input<VDBFloatGrid>("VDBGrid")->m_grid;
    else
      gridPtr =
          zs::load_floatgrid_from_vdb_file(get_param<std::string>("path"));

    // auto spls = zs::convert_floatgrid_to_sparse_levelset(
    //    gridPtr, {zs::memsrc_e::host, -1});
    auto dx = get_input2<float>("dx");
#if 0
    auto sampled = zs::sample_from_levelset(
        zs::proxy<zs::execspace_e::openmp>(spls), dx, get_input2<float>("ppc"));
#else
    auto sampled =
        zs::sample_from_levelset(gridPtr, dx, get_input2<float>("ppc"));
#endif

    auto prim = std::make_shared<PrimitiveObject>();
    prim->resize(sampled.size());
    auto &pos = prim->attr<vec3f>("pos");
    auto &vel = prim->add_attr<vec3f>("vel");
    // auto &nrm = prim->add_attr<vec3f>("nrm");

    /// compute default normal
    auto ompExec = zs::omp_exec();
#if 0
    const auto calcNormal = [spls = proxy<zs::execspace_e::host>(spls),
                             eps = dx](const vec3f &x_) {
      zs::vec<float, 3> x{x_[0], x_[1], x_[2]}, diff{};
      /// compute a local partial derivative
      for (int i = 0; i != 3; i++) {
        auto v1 = x;
        auto v2 = x;
        v1[i] = x[i] + eps;
        v2[i] = x[i] - eps;
        diff[i] = (spls.getSignedDistance(v1) - spls.getSignedDistance(v2)) /
                  (eps + eps);
      }
      if (math::near_zero(diff.l2NormSqr()))
        return vec3f{0, 0, 0};
      auto r = diff.normalized();
      return vec3f{r[0], r[1], r[2]};
    };
#endif
    ompExec(zs::range(sampled.size()), [&sampled, &pos, &vel](size_t pi) {
      pos[pi] = sampled[pi];
      vel[pi] = vec3f{0, 0, 0};
      // nrm[pi] = calcNormal(pos[pi]);
    });

    fmt::print(fg(fmt::color::cyan), "done executing PoissonDiskSample\n");
    set_output("prim", std::move(prim));
  }
};
ZENDEFNODE(PoissonDiskSample,
           {
               {"VDBGrid", {"float", "dx", "0.1"}, {"float", "ppc", "8"}},
               {"prim"},
               {{"string", "path", ""}},
               {"MPM"},
           });

struct ZSPoissonDiskSample : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green), "begin executing ZSPoissonDiskSample\n");
    auto prim = std::make_shared<PrimitiveObject>();
    std::vector<std::array<float, 3>> sampled;

    auto zsfield = get_input<ZenoLevelSet>("ZSLevelSet");
    auto dx = get_input2<float>("dx");
    auto ppc = get_input2<float>("ppc");

    match([&prim, &sampled, dx, ppc, this](auto &ls) {
      using LsT = RM_CVREF_T(ls);
      if constexpr (is_same_v<LsT, typename ZenoLevelSet::basic_ls_t>) {
        auto &field = ls._ls;
        match(
            [&sampled = sampled, dx = dx, ppc = ppc](auto &lsPtr)
                -> std::enable_if_t<
                    is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
              const auto &ls = *lsPtr;
              const auto &spls = ls.memspace() != memsrc_e::host
                                     ? ls.clone({memsrc_e::host, -1})
                                     : ls;
              sampled = zs::sample_from_levelset(
                  zs::proxy<zs::execspace_e::openmp>(spls), dx, ppc);
            },
            [](auto &lsPtr)
                -> std::enable_if_t<
                    !is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
              throw std::runtime_error(
                  fmt::format("levelset type [{}] not supported in sampling.",
                              zs::get_var_type_str(lsPtr)));
            })(field);
      } else if constexpr (is_same_v<
                               LsT,
                               typename ZenoLevelSet::const_sdf_vel_ls_t>) {
        match([&sampled = sampled, dx = dx, ppc = ppc](auto lsv) {
          sampled = zs::sample_from_levelset(SdfVelFieldView{lsv}, dx, ppc);
        })(ls.template getView<execspace_e::openmp>());
      } else if constexpr (is_same_v<
                               LsT,
                               typename ZenoLevelSet::const_transition_ls_t>) {
        match([&sampled = sampled, &ls, dx = dx, ppc = ppc](auto fieldPair) {
          sampled = zs::sample_from_levelset(
              TransitionLevelSetView{SdfVelFieldView{get<0>(fieldPair)},
                                     SdfVelFieldView{get<1>(fieldPair)},
                                     ls._stepDt, ls._alpha},
              dx, ppc);
        })(ls.template getView<execspace_e::openmp>());
      } else {
        throw std::runtime_error("unknown levelset type...");
      }
    })(zsfield->levelset);

    prim->resize(sampled.size());
    auto &pos = prim->attr<vec3f>("pos");
    auto &vel = prim->add_attr<vec3f>("vel");

    /// compute default normal
    auto ompExec = zs::omp_exec();
    ompExec(zs::range(sampled.size()), [&sampled, &pos, &vel](size_t pi) {
      pos[pi] = sampled[pi];
      vel[pi] = vec3f{0, 0, 0};
      // nrm[pi] = calcNormal(pos[pi]);
    });

    fmt::print(fg(fmt::color::cyan), "done executing ZSPoissonDiskSample\n");
    set_output("prim", std::move(prim));
  }
};
ZENDEFNODE(ZSPoissonDiskSample,
           {
               {"ZSLevelSet", {"float", "dx", "0.1"}, {"float", "ppc", "8"}},
               {"prim"},
               {{"string", "path", ""}},
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
        using TV3 = zs::vec<double, 3>;
        const auto triArea = [&obj](vec3i tri) {
          TV3 p0 = TV3{obj[tri[0]][0], obj[tri[0]][1], obj[tri[0]][2]};
          TV3 p1 = TV3{obj[tri[1]][0], obj[tri[1]][1], obj[tri[1]][2]};
          TV3 p2 = TV3{obj[tri[2]][0], obj[tri[2]][1], obj[tri[2]][2]};
          return (p1 - p0).cross(p2 - p0).norm() * 0.5;
          // return length(cross(obj[tri[1]] - p0, obj[tri[2]] - p0)) * 0.5;
        };
        const auto toTV3 = [](const zeno::vec3f &x) {
          return TV3{x[0], x[1], x[2]};
        };
        const auto fromTV3 = [](const TV3 &x) {
          return zeno::vec3f{(float)x[0], (float)x[1], (float)x[2]};
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
          eleVol[i] = std::max(v, limits<float>::epsilon() * 10.);
          elePos[i] = (obj[tri[0]] + obj[tri[1]] + obj[tri[2]]) / 3;
          if (velsPtr)
            eleVel[i] =
                (velsPtr[tri[0]] + velsPtr[tri[1]] + velsPtr[tri[2]]) / 3;
          eleD[i][0] = obj[tri[1]] - obj[tri[0]];
          eleD[i][1] = obj[tri[2]] - obj[tri[0]];

          auto normal = cross(toTV3(eleD[i][1]) / model->dx,
                              toTV3(eleD[i][0]) / model->dx)
                            .normalized();
          eleD[i][2] = fromTV3(normal);

          for (auto pi : tri)
            dofVol[pi] += std::max(v, limits<float>::epsilon() * 10.) / 3;
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
        {"mass", 1}, {"pos", 3},   {"vel", 3},
        {"vol", 1},  {"C", 9},     {"F", 9},
        {"d", 9},    {"DmInv", 9}, {"inds", (int)category + 1}};

    const bool hasLogJp = model->hasLogJp();
    const bool hasOrientation = model->hasOrientation();
    const bool hasF = model->hasF();

    if (!bindMesh) {
      if (hasF)
        tags.emplace_back(zs::PropertyTag{"F", 9});
      else {
        tags.emplace_back(zs::PropertyTag{"J", 1});
        if (category != ZenoParticles::mpm)
          throw std::runtime_error(
              "mesh particles should not use the 'J' attribute.");
      }
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
                                velsPtr, nrmsPtr, bindMesh, &dofVol, category,
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
        if (!bindMesh) {
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
            // pars.tuple<3>("a", pi) = vec3::zeros();
            pars.tuple<3>("a", pi) = vec3{0, 1, 0};
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
               nrmsPtr, &eleVol, &elePos, &obj, &eleVel, &eleD, category,
               &quads, &tris, &lines](size_t ei) mutable {
                using vec3 = zs::vec<double, 3>;
                using mat3 = zs::vec<double, 3, 3>;
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
                eles.tuple<9>("d", ei) = Dmat;
                // ref: CFF Jiang, 2017 Anisotropic MPM techdoc
                // ref: Yun Fei, libwetcloth;
                auto t0 = col(Dmat, 0);
                auto t1 = col(Dmat, 1);
                auto normal = col(Dmat, 2);
                // could qr decomp here first (tech doc)

                zs::Rotation<double, 3> rot0{normal, vec3{0, 0, 1}};
                auto u = rot0 * t0;
                auto v = rot0 * t1;
                zs::Rotation<double, 3> rot1{u, vec3{1, 0, 0}};
                auto ru = rot1 * u;
                auto rv = rot1 * v;
                auto Dstar = mat3::identity();
                Dstar(0, 0) = ru(0);
                Dstar(0, 1) = rv(0);
                Dstar(1, 1) = rv(1);

                if (std::abs(rv(1)) < 10 * limits<float>::epsilon() ||
                    std::abs(ru(0)) < 10 * limits<float>::epsilon()) {
                  eles.tuple<9>("DmInv", ei) = mat3::identity();
                  eles.tuple<9>("F", ei) = Dmat;
                  puts("beware: encounters near-singular Dm element");
                } else {
                  auto invDstar = zs::inverse(Dstar);
                  eles.tuple<9>("DmInv", ei) = invDstar;
                  eles.tuple<9>("F", ei) = Dmat * invDstar;
                }

                // apic transfer
                eles.tuple<9>("C", ei) = mat3::zeros();

                // plasticity

                // element-vertex indices
                if (category == ZenoParticles::tet) {
                  const auto &quad = quads[ei];
                  for (int i = 0; i != 4; ++i) {
                    eles("inds", i, ei) = reinterpret_bits<float>(quad[i]);
                  }
                } else if (category == ZenoParticles::surface) {
                  const auto &tri = tris[ei];
                  for (int i = 0; i != 3; ++i) {
                    eles("inds", i, ei) = reinterpret_bits<float>(tri[i]);
                  }
                } else if (category == ZenoParticles::curve) {
                  const auto &line = lines[ei];
                  for (int i = 0; i != 2; ++i) {
                    eles("inds", i, ei) = reinterpret_bits<float>(line[i]);
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

struct ScalePrimitiveAlongNormal : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    set_output("prim", get_input("prim"));

    if (!prim->has_attr("nrm"))
      return;

    auto &nrm = prim->attr<zeno::vec3f>("nrm");
    auto &pos = prim->attr<zeno::vec3f>("pos");
    auto dis = get_input2<float>("dis");

    auto ompExec = zs::omp_exec();
    ompExec(zs::range(pos.size()),
            [&](size_t pi) { pos[pi] += nrm[pi] * dis; });
  }
};

ZENDEFNODE(ScalePrimitiveAlongNormal, {
                                          {"prim", {"float", "dis", "0"}},
                                          {"prim"},
                                          {},
                                          {"primitive"},
                                      });

struct ComputePrimitiveSequenceVelocity : zeno::INode {
  virtual void apply() override {
    auto prim0 = get_input<PrimitiveObject>("prim0");
    auto prim1 = get_input<PrimitiveObject>("prim1");

    if (prim0->size() != prim1->size())
      throw std::runtime_error(
          "consecutive sequence objs with different topo!");

    auto &p0 = prim0->attr<zeno::vec3f>("pos");
    auto &p1 = prim1->attr<zeno::vec3f>("pos");
    auto &v0 = prim0->add_attr<zeno::vec3f>("vel");
    auto &v1 = prim1->add_attr<zeno::vec3f>("vel");

    auto ompExec = zs::omp_exec();
    ompExec(zs::range(p0.size()), [&, dt = get_input2<float>("dt")](size_t pi) {
      v0[pi] = (p1[pi] - p0[pi]) / dt;
      v1[pi] = vec3f{0, 0, 0};
    });
  }
};

ZENDEFNODE(ComputePrimitiveSequenceVelocity,
           {
               {"prim0", "prim1", {"float", "dt", "1"}},
               {},
               {},
               {"primitive"},
           });

struct ToZSLevelSet : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZSLevelSet\n");
    auto ls = std::make_shared<ZenoLevelSet>();

    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;

    if (has_input<VDBFloatGrid>("VDBGrid")) {
      // pass in FloatGrid::Ptr
      zs::OpenVDBStruct gridPtr = get_input<VDBFloatGrid>("VDBGrid")->m_grid;
      ls->getLevelSet() = basic_ls_t{zs::convert_floatgrid_to_sparse_levelset(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0})};
    } else if (has_input<VDBFloat3Grid>("VDBGrid")) {
      // pass in FloatGrid::Ptr
      zs::OpenVDBStruct gridPtr = get_input<VDBFloat3Grid>("VDBGrid")->m_grid;
      ls->getLevelSet() =
          basic_ls_t{zs::convert_vec3fgrid_to_sparse_staggered_grid(
              gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0})};
    } else {
      auto path = get_param<std::string>("path");
      auto gridPtr = zs::load_vec3fgrid_from_vdb_file(path);
      ls->getLevelSet() = basic_ls_t{zs::convert_vec3fgrid_to_sparse_levelset(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0})};
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
    auto ls = std::make_shared<ZenoLevelSet>();

    std::shared_ptr<ZenoLevelSet> sdfLsPtr{};
    std::shared_ptr<ZenoLevelSet> velLsPtr{};

    using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;

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
      ls->getLevelSet() = const_sdf_vel_ls_t{sdfLsPtr->getBasicLevelSet(),
                                             velLsPtr->getBasicLevelSet()};
    } else {
      if (!sdfLsPtr->holdsBasicLevelSet()) {
        auto msg = fmt::format("sdfField is {}a basic levelset.\n",
                               sdfLsPtr->holdsBasicLevelSet() ? "" : "not ");
        throw std::runtime_error(msg);
      }
      ls->getLevelSet() = const_sdf_vel_ls_t{sdfLsPtr->getBasicLevelSet()};
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
    using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
    using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;

    std::shared_ptr<ZenoLevelSet> zsls{};
    if (has_input<ZenoLevelSet>("ZSLevelSetSequence"))
      zsls = get_input<ZenoLevelSet>("ZSLevelSetSequence");
    else {
      zsls = std::make_shared<ZenoLevelSet>();
      zsls->levelset = const_transition_ls_t{};
    }
    auto &lsseq = zsls->getLevelSetSequence();

    if (has_input<ZenoLevelSet>("ZSLevelSet")) {
      auto &ls = get_input<ZenoLevelSet>("ZSLevelSet")->getLevelSet();
      match(
          [&lsseq](basic_ls_t &basicLs) {
            lsseq.push(const_sdf_vel_ls_t{basicLs});
          },
          [&lsseq](const_sdf_vel_ls_t &field) { // recommend
            lsseq.push(field); // also reset alpha in the meantime
          },
          [&lsseq](const_transition_ls_t &seq) {
            lsseq._fields.insert(lsseq._fields.end(), seq._fields.begin(),
                                 seq._fields.end());
          })(ls);
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

struct MakeZSString : INode {
  void apply() override {
    auto n = std::make_shared<StringObject>();
    n->value = fmt::format(get_input<StringObject>("fmt_str")->get(),
                           get_input<NumericObject>("frameid")->get<int>(),
                           get_input<NumericObject>("stepid")->get<int>());
    set_output("str", std::move(n));
  }
};
ZENDEFNODE(MakeZSString, {
                             {"fmt_str", "frameid", "stepid"},
                             {"str"},
                             {},
                             {"SOP"},
                         });

/// update levelsetsequence state
struct UpdateLevelSetSequence : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing UpdateLevelSetSequence\n");

    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
    using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
    using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;

    std::shared_ptr<ZenoLevelSet> zsls{};
    if (has_input<ZenoLevelSet>("ZSLevelSetSequence"))
      zsls = get_input<ZenoLevelSet>("ZSLevelSetSequence");
    else {
      zsls = std::make_shared<ZenoLevelSet>();
      zsls->getLevelSet() = const_transition_ls_t{};
    }
    auto &lsseq = zsls->getLevelSetSequence();

    if (has_input<NumericObject>("dt")) {
      auto stepDt = get_input<NumericObject>("dt")->get<float>();
      lsseq.setStepDt(stepDt);
      // fmt::print("\tdt: {}\n", lsseq._stepDt);
    }

    if (has_input<NumericObject>("alpha")) {
      auto alpha = get_input<NumericObject>("alpha")->get<float>();
      lsseq.advance(alpha);
      // fmt::print("\talpha: {}, accum: {}\n", alpha, lsseq._alpha);
    }

#if 0
    fmt::print("\t{} levelset in queue.\n", lsseq._fields.size());
    match([&lsseq](auto fieldPair) {
      auto lsseqv = TransitionLevelSetView{SdfVelFieldView{get<0>(fieldPair)},
                                           SdfVelFieldView{get<1>(fieldPair)},
                                           lsseq._stepDt, lsseq._alpha};
      fmt::print("ls_seq_view type: [{}]\n", get_var_type_str(lsseqv));
      auto printBox = [](std::string_view msg, auto &lsv) {
        auto [mi, ma] = lsv.getBoundingBox();
        fmt::print("[{}]: [{}, {}, {} ~ {}, {}, {}]\n", msg, mi[0], mi[1],
                   mi[2], ma[0], ma[1], ma[2]);
      };
      printBox("src", lsseqv._lsvSrc);
      printBox("dst", lsseqv._lsvDst);
      printBox("cur", lsseqv);
    })(lsseq.template getView<execspace_e::openmp>());
#endif

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
    auto vdb = std::make_shared<VDBFloatGrid>();

    if (has_input<ZenoLevelSet>("ZSLevelSet")) {
      auto ls = get_input<ZenoLevelSet>("ZSLevelSet");
      if (ls->holdsBasicLevelSet()) {
        zs::match(
            [&vdb](auto &lsPtr)
                -> std::enable_if_t<
                    zs::is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
              using LsT = typename RM_CVREF_T(lsPtr)::element_type;
              vdb->m_grid = zs::convert_sparse_levelset_to_vdbgrid(*lsPtr)
                                .template as<openvdb::FloatGrid::Ptr>();
            },
            [](...) {})(ls->getBasicLevelSet()._ls);
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