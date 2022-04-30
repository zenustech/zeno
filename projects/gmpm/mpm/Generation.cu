#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>

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
      if (auto it = params->lut.find(tag); it != params->lut.end()) {
        if (auto p = std::dynamic_pointer_cast<NumericObject>(it->second)) {
          ret = p->get<T>();
        }
      }
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

struct ToTrackerParticles : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToTrackerParticles\n");

    // primitive
    auto inParticles = get_input<PrimitiveObject>("prim");
    auto &obj = inParticles->attr<vec3f>("pos");
    vec3f *velsPtr{nullptr};
    if (inParticles->has_attr("vel"))
      velsPtr = inParticles->attr<vec3f>("vel").data();

    auto outParticles = std::make_shared<ZenoParticles>();

    // primitive binding
    outParticles->prim = inParticles;

    /// category, size
    std::size_t size{obj.size()};
    outParticles->category = ZenoParticles::category_e::tracker;

    // per vertex (node) vol, pos, vel
    using namespace zs;
    auto ompExec = zs::omp_exec();

    // attributes
    std::vector<zs::PropertyTag> tags{{"pos", 3}, {"vel", 3}};
    {
      auto &pars = outParticles->getParticles(); // tilevector
      pars = typename ZenoParticles::particles_t{tags, size, memsrc_e::host};
      ompExec(zs::range(size), [pars = proxy<execspace_e::host>({}, pars),
                                velsPtr, &obj](size_t pi) mutable {
        using vec3 = zs::vec<float, 3>;
        using mat3 = zs::vec<float, 3, 3>;

        // pos
        pars.tuple<3>("pos", pi) = obj[pi];

        // vel
        if (velsPtr != nullptr)
          pars.tuple<3>("vel", pi) = velsPtr[pi];
        else
          pars.tuple<3>("vel", pi) = vec3::zeros();
      });

      pars = pars.clone({memsrc_e::um, 0});
    }
    if (inParticles->tris.size()) {
      const auto eleSize = inParticles->tris.size();
      std::vector<zs::PropertyTag> tags{{"pos", 3}, {"vel", 3}, {"inds", 3}};
      outParticles->elements =
          typename ZenoParticles::particles_t{tags, eleSize, memsrc_e::host};
      auto &eles = outParticles->getQuadraturePoints();

      auto &tris = inParticles->tris.values;
      ompExec(zs::range(eleSize), [eles = proxy<execspace_e::host>({}, eles),
                                   &obj, &tris, velsPtr](size_t ei) mutable {
        using vec3 = zs::vec<float, 3>;
        // inds
        int inds[3] = {(int)tris[ei][0], (int)tris[ei][1], (int)tris[ei][2]};
        for (int d = 0; d != 3; ++d)
          eles("inds", d, ei) = reinterpret_bits<float>(inds[d]);
        // pos
        eles.tuple<3>("pos", ei) =
            (obj[inds[0]] + obj[inds[1]] + obj[inds[2]]) / 3.f;

        // vel
        if (velsPtr != nullptr) {
          eles.tuple<3>("vel", ei) =
              (velsPtr[inds[0]] + velsPtr[inds[1]] + velsPtr[inds[2]]) / 3.f;
        } else
          eles.tuple<3>("vel", ei) = vec3::zeros();
      });

      eles = eles.clone({memsrc_e::um, 0});
    }

    fmt::print(fg(fmt::color::cyan), "done executing ToTrackerParticles\n");
    set_output("ZSParticles", outParticles);
  }
};

ZENDEFNODE(ToTrackerParticles, {
                                   {"prim"},
                                   {"ZSParticles"},
                                   {},
                                   {"MPM"},
                               });

struct ConstructBendingSprings : INode {
  std::shared_ptr<ZenoParticles>
  addBendingSprings(zs::CudaExecutionPolicy &cudaPol, const ZenoParticles &surf,
                    float stiffness) {
    if (surf.category != ZenoParticles::surface)
      return {};
    using namespace zs;
    auto outParticles = std::make_shared<ZenoParticles>();
    using TableT = HashTable<int, 2, int>;     //
    using VertTableT = HashTable<int, 1, int>; //
    using key_t = typename TableT::key_t;
    using vec1i = zs::vec<int, 1>;
    using vec3 = zs::vec<float, 3>;
    using mat3 = zs::vec<float, 3, 3>;
    auto &surfPars = surf.getParticles();
    auto numV = surfPars.size(); // i.e. sprayedOffset
    auto &surfEles = surf.getQuadraturePoints();
    auto numE = surfEles.size();

    fmt::print("surface mesh: {} verts, {} tris.\n", numV, numE);
    TableT edgeTable{surfPars.get_allocator(), numE * 3}; // edge -> eleid
    edgeTable.reset(cudaPol, true);
    //
    constexpr auto space = execspace_e::cuda;
    cudaPol(range(numE),
            [table = proxy<space>(edgeTable),
             eles = proxy<space>({}, surfEles)] __device__(int ei) mutable {
              auto tri =
                  eles.pack<3>("inds", ei).template reinterpret_bits<int>();
              auto vi = tri[2];
              for (int v = 0; v != 3; ++v) {
                auto vj = tri[v];
                if (vi < vj)
                  table.insert(key_t{vi, vj});
                vi = vj;
              }
            });
    std::size_t numRegisteredEdges = edgeTable.size();
    Vector<int> edgeToEles{surfPars.get_allocator(), numRegisteredEdges};
    cudaPol(
        range(numE),
        [table = proxy<space>(edgeTable), edgeToEles = proxy<space>(edgeToEles),
         eles = proxy<space>({}, surfEles)] __device__(int ei) mutable {
          auto tri = eles.pack<3>("inds", ei).template reinterpret_bits<int>();
          auto vi = tri[2];
          for (int v = 0; v != 3; ++v) {
            auto vj = tri[v];
            if (vi < vj) {
              auto no = table.query(key_t{vi, vj});
              edgeToEles[no] = ei;
            }
            vi = vj;
          }
        });
    //
    using VertPair = zs::vec<int, 2>;
    Vector<int> cnt{surfPars.get_allocator(), 1};
    cnt.setVal(0);
    Vector<VertPair> vertPairs{surfPars.get_allocator(), numRegisteredEdges};
    Vector<VertPair> elePairs{surfPars.get_allocator(), numRegisteredEdges};
    VertTableT vertTable{surfPars.get_allocator(), numRegisteredEdges * 2};
    vertTable.reset(cudaPol, true);
    cudaPol(
        range(numE),
        [table = proxy<space>(edgeTable), vertTable = proxy<space>(vertTable),
         edgeToEles = proxy<space>(edgeToEles), cnt = proxy<space>(cnt),
         vertPairs = proxy<space>(vertPairs), elePairs = proxy<space>(elePairs),
         eles = proxy<space>({}, surfEles)] __device__(int ei) mutable {
          using table_t = RM_CVREF_T(table);
          auto tri = eles.pack<3>("inds", ei).template reinterpret_bits<int>();
          auto vi = tri[2];
          for (int v = 0; v != 3; ++v) {
            auto vj = tri[v];
            if (vi > vj) { // check opposite
              if (auto edgeNo = table.query(key_t{vj, vi});
                  edgeNo != table_t::sentinel_v) {
                auto neighborEleNo = edgeToEles[edgeNo];
                auto neighborTri = eles.pack<3>("inds", neighborEleNo)
                                       .template reinterpret_bits<int>();
                int neighborV = -1, selfV = -1;
                for (int d = 0; d != 3; ++d)
                  if (neighborTri[d] != vi && neighborTri[d] != vj) {
                    neighborV = neighborTri[d];
                    break;
                  }
                for (int d = 0; d != 3; ++d)
                  if (tri[d] != vi && tri[d] != vj) {
                    selfV = tri[d];
                    break;
                  }
                vertTable.insert(vec1i{neighborV});
                vertTable.insert(vec1i{selfV});
                auto no = atomic_add(exec_cuda, &cnt[0], 1);
                vertPairs[no] = VertPair{neighborV, selfV};
                elePairs[no] = VertPair{neighborEleNo, ei};
              }
            }
            vi = vj;
          }
        });
    std::size_t numVertPairs = cnt.getVal();
    vertPairs.resize(numVertPairs);
    elePairs.resize(numVertPairs);
    //
    auto ret = std::make_shared<ZenoParticles>();
    ret->sprayedOffset = cnt.getVal();
    ret->getModel() = surf.getModel();
    ret->category = ZenoParticles::curve;

    std::vector<zs::PropertyTag> tags{{"mass", 1}, {"pos", 3}, {"vel", 3},
                                      {"vol", 1},  {"C", 9},   {"beta", 1}};
    std::vector<zs::PropertyTag> eleTags{
        {"mass", 1},      {"pos", 3}, {"vel", 3}, {"vol", 1},
        {"C", 9},         {"F", 9},   {"d", 9},   {"DmInv", 9},
        {"inds", (int)2}, {"mu", 1},  {"lam", 1}};

    auto &pars = ret->getParticles(); // tilevector
    std::size_t numSpringVerts = vertTable.size();
    pars = typename ZenoParticles::particles_t{surfPars.get_allocator(), tags,
                                               numSpringVerts};
    // springs have no inertial
    cudaPol(range(numSpringVerts),
            [pars = proxy<space>({}, pars),
             surfPars = proxy<space>({}, surfPars),
             vertTable = proxy<space>(vertTable)] __device__(int pi) mutable {
              using mat3 = zs::vec<float, 3, 3>;
              auto opid = vertTable._activeKeys[pi][0];
              pars("mass", pi) = 0.f;
              pars("vol", pi) = surfPars("vol", opid);
              pars("beta", pi) = 0.f;
              pars.tuple<3>("pos", pi) = surfPars.pack<3>("pos", opid);
              pars.tuple<3>("vel", pi) = vec3::zeros();
              pars.tuple<3 * 3>("C", pi) = mat3::zeros();
            });

    ret->elements = typename ZenoParticles::particles_t{
        surfPars.get_allocator(), eleTags, numVertPairs};
    auto &eles = ret->getQuadraturePoints();
    cudaPol(range(numVertPairs), [pars = proxy<space>({}, pars),
                                  eles = proxy<space>({}, eles),
                                  surfEles = proxy<space>({}, surfEles),
                                  vertPairs = proxy<space>(vertPairs),
                                  elePairs = proxy<space>(elePairs),
                                  vertTable = proxy<space>(vertTable),
                                  stiffness] __device__(int ei) mutable {
      using mat3 = zs::vec<float, 3, 3>;
      eles("mass", ei) = 0.f;

      {
        auto eids = elePairs[ei];
        auto mu = zs::min(surfEles("mu", eids[0]), surfEles("mu", eids[1]));
        auto lam = zs::min(surfEles("lam", eids[0]), surfEles("lam", eids[1]));
        eles("mu", ei) = mu * stiffness;
        eles("lam", ei) = lam * stiffness;
      }

      auto inds = vertPairs[ei];
      inds[0] = vertTable.query(vec1i{inds[0]});
      inds[1] = vertTable.query(vec1i{inds[1]});
      vec3 xs[2];
      xs[0] = pars.pack<3>("pos", inds[0]);
      xs[1] = pars.pack<3>("pos", inds[1]);
      eles.tuple<3>("pos", ei) = (xs[0] + xs[1]) / 2;
      eles("vol", ei) = (pars("vol", inds[0]) + pars("vol", inds[1])) / 2;
      eles.tuple<3>("vel", ei) = vec3::zeros();

      eles.tuple<3 * 3>("C", ei) = mat3::zeros();

      auto tangent = xs[1] - xs[0];
      auto tn = tangent.norm();
      auto nrm = tangent.orthogonal().normalized();
      auto binrm = tangent.cross(nrm).normalized();
      auto d = mat3{tangent[0], nrm[0],     binrm[0], tangent[1], nrm[1],
                    binrm[1],   tangent[2], nrm[2],   binrm[2]};
      eles.tuple<3 * 3>("d", ei) = d;
      auto invDstar = mat3::identity();
      invDstar(0, 0) = 1. / tn;
      if (tn <= 10 * limits<float>::epsilon()) {
        eles("mu", ei) = 0.f;
        eles("lam", ei) = 0.f;
      }
      eles.tuple<3 * 3>("DmInv", ei) = invDstar;
      eles.tuple<3 * 3>("F", ei) = d * invDstar;

      eles.tuple<2>("inds", ei) = inds.template reinterpret_bits<float>();
    });

    fmt::print("bending spring mesh: {} verts, {} tris.\n", numSpringVerts,
               numVertPairs);
    return ret;
  }
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing ConstructBendingSprings\n");

    float stiffness = get_input2<float>("bending_stiffness");
    auto cudaPol = cuda_exec();
    if (has_input<ZenoParticles>("ZSSurfPrim")) {
      set_output("ZSSpringPrim",
                 addBendingSprings(cudaPol,
                                   *get_input<ZenoParticles>("ZSSurfPrim"),
                                   stiffness));
    } else if (has_input<ListObject>("ZSSurfPrim")) {
      auto list = std::make_shared<ListObject>();
      auto &ret = list->arr;
      auto &objSharedPtrLists = *get_input<zeno::ListObject>("ZSSurfPrim");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          ret.push_back(addBendingSprings(cudaPol, *ptr, stiffness));
      set_output("ZSSpringPrim", list);
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing ConstructBendingSprings\n");
  }
};

ZENDEFNODE(ConstructBendingSprings,
           {
               {"ZSSurfPrim", {"float", "bending_stiffness", "0.01"}},
               {"ZSSpringPrim"},
               {},
               {"MPM"},
           });

struct BuildPrimitiveSequence : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing BuildPrimitiveSequence\n");

    std::shared_ptr<ZenoParticles> zsprimseq{};

    if (!has_input<ZenoParticles>("ZSParticles"))
      throw std::runtime_error(
          fmt::format("no incoming prim for prim sequence!\n"));
    auto next = get_input<ZenoParticles>("ZSParticles");
    if (!next->asBoundary)
      throw std::runtime_error(
          fmt::format("incoming prim is not used as a boundary!\n"));

    auto cudaPol = cuda_exec().device(0);
    if (has_input<ZenoParticles>("ZSPrimitiveSequence")) {
      zsprimseq = get_input<ZenoParticles>("ZSPrimitiveSequence");
      auto numV = zsprimseq->numParticles();
      auto numE = zsprimseq->numElements();
      auto sprayedOffset = zsprimseq->sprayedOffset;
      auto sprayedSize = numV - sprayedOffset;
      auto size = sprayedOffset;
      if (size != next->numParticles() || numE != next->numElements()) {
        fmt::print(
            "current numVerts ({} + {}) (i.e. {}), numEles ({}).\nIncoming "
            "boundary primitive numVerts ({}), numEles ({})\n",
            size, sprayedSize, numV, numE, next->numParticles(),
            next->numElements());
        throw std::runtime_error(
            fmt::format("prim size mismatch with current sequence prim!\n"));
      }

      fmt::print("{} verts (including {} sprayed), {} elements\n", numV,
                 sprayedSize, numE);

      auto dt = get_input2<float>("framedt"); // framedt
      /// update velocity
      // update mesh verts
      cudaPol(Collapse{size},
              [prev = proxy<execspace_e::cuda>({}, zsprimseq->getParticles()),
               next = proxy<execspace_e::cuda>({}, next->getParticles()),
               dt] __device__(int pi) mutable {
                prev.tuple<3>("vel", pi) =
                    (next.pack<3>("pos", pi) - prev.pack<3>("pos", pi)) / dt;
              });
      // update elements
      cudaPol(Collapse{numE},
              [prev = proxy<execspace_e::cuda>(
                   {}, zsprimseq->getQuadraturePoints()),
               next = proxy<execspace_e::cuda>({}, next->getQuadraturePoints()),
               dt] __device__(int ei) mutable {
                prev.tuple<3>("vel", ei) =
                    (next.pack<3>("pos", ei) - prev.pack<3>("pos", ei)) / dt;
              });
      if (size != numV) { // update sprayed mesh verts
        cudaPol(
            Collapse{sprayedSize},
            [verts = proxy<execspace_e::cuda>({}, zsprimseq->getParticles()),
             eles =
                 proxy<execspace_e::cuda>({}, zsprimseq->getQuadraturePoints()),
             sprayedOffset] __device__(int pi) mutable {
              auto dst = pi + sprayedOffset;

              int eid = reinterpret_bits<int>(verts("eid", dst));
              auto tri = eles.pack<3>("inds", eid).reinterpret_bits<int>();
              auto ws = verts.pack<3>("weights", dst);
              {
                auto v0 = verts.pack<3>("vel", tri[0]);
                auto v1 = verts.pack<3>("vel", tri[1]);
                auto v2 = verts.pack<3>("vel", tri[2]);

                verts.tuple<3>("vel", dst) =
                    ws[0] * v0 + ws[1] * v1 + ws[2] * v2;
              }
              {
                auto p0 = verts.pack<3>("pos", tri[0]);
                auto p1 = verts.pack<3>("pos", tri[1]);
                auto p2 = verts.pack<3>("pos", tri[2]);

                verts.tuple<3>("pos", dst) =
                    ws[0] * p0 + ws[1] * p1 + ws[2] * p2;
              }
            });
      }
    } else {
      zsprimseq = std::make_shared<ZenoParticles>(*next);
    }

    fmt::print(fg(fmt::color::cyan), "done executing BuildPrimitiveSequence\n");
    set_output("ZSPrimitiveSequence", zsprimseq);
  }
};
ZENDEFNODE(BuildPrimitiveSequence, {
                                       {"ZSPrimitiveSequence",
                                        {"float", "framedt", "0.1"},
                                        "ZSParticles"},
                                       {"ZSPrimitiveSequence"},
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

      auto &prim = *parObjPtr->prim;
      // const auto category = parObjPtr->category;
      auto &pos = prim.attr<vec3f>("pos");
      auto size = pos.size(); // in case zsparticle-mesh is refined
      vec3f *velsPtr{nullptr};
      if (prim.has_attr("vel") && pars.hasProperty("vel"))
        velsPtr = prim.attr<vec3f>("vel").data();

      if (pars.hasProperty("id")) {
        ompExec(range(pars.size()),
                [&, pars = proxy<execspace_e::host>({}, pars)](auto pi) {
                  auto id = (int)pars("id", pi);
                  if (id >= size)
                    return;
                  pos[id] = pars.array<3>("pos", pi);
                  if (velsPtr != nullptr)
                    velsPtr[id] = pars.array<3>("vel", pi);
                });
      } else {
        // currently only write back pos and vel (if exists)
        ompExec(range(size),
                [&, pars = proxy<execspace_e::host>({}, pars)](auto pi) {
                  pos[pi] = pars.array<3>("pos", pi);
                  if (velsPtr != nullptr)
                    velsPtr[pi] = pars.array<3>("vel", pi);
                });
      }
      const auto cnt = pars.size();
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
    partition->get() = typename ZenoPartition::table_t{(std::size_t)1,
                                                       zs::memsrc_e::device, 0};
    partition->requestRebuild = false;
    partition->rebuilt = false;
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
      tags.emplace_back(zs::PropertyTag{"vstar", 3});
    else if (grid->transferScheme == "apic")
      ;
    else if (grid->transferScheme == "aflip")
      tags.emplace_back(zs::PropertyTag{"vstar", 3});
    else if (grid->transferScheme == "boundary")
      tags.emplace_back(zs::PropertyTag{"nrm", 3});
    else
      throw std::runtime_error(fmt::format(
          "unrecognized transfer scheme [{}]\n", grid->transferScheme));

    grid->get() =
        typename ZenoGrid::grid_t{tags, dx, 1, zs::memsrc_e::device, 0};

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
    ls->transferScheme = get_param<std::string>("transfer");
    auto cateStr = get_param<std::string>("category");

    // default is "cellcentered"
    if (cateStr == "staggered")
      tags.emplace_back(zs::PropertyTag{"vel", 3});
    // default is "unknown"
    if (ls->transferScheme == "unknown")
      ;
    else if (ls->transferScheme == "flip")
      tags.emplace_back(zs::PropertyTag{"vstar", 3});
    else if (ls->transferScheme == "apic")
      ;
    else if (ls->transferScheme == "aflip")
      tags.emplace_back(zs::PropertyTag{"vstar", 3});
    else if (ls->transferScheme == "boundary")
      tags.emplace_back(zs::PropertyTag{"nrm", 3});
    else
      throw std::runtime_error(fmt::format(
          "unrecognized transfer scheme [{}]\n", ls->transferScheme));

    if (cateStr == "collocated") {
      auto tmp = typename ZenoLevelSet::template spls_t<zs::grid_e::collocated>{
          tags, dx, 1, zs::memsrc_e::device, 0};
      tmp.reset(zs::cuda_exec(), 0);
      ls->getLevelSet() = std::move(tmp);
    } else if (cateStr == "cellcentered") {
      auto tmp =
          typename ZenoLevelSet::template spls_t<zs::grid_e::cellcentered>{
              tags, dx, 1, zs::memsrc_e::device, 0};
      tmp.reset(zs::cuda_exec(), 0);
      ls->getLevelSet() = std::move(tmp);
    } else if (cateStr == "staggered") {
      auto tmp = typename ZenoLevelSet::template spls_t<zs::grid_e::staggered>{
          tags, dx, 1, zs::memsrc_e::device, 0};
      tmp.reset(zs::cuda_exec(), 0);
      ls->getLevelSet() = std::move(tmp);
    } else if (cateStr == "const_velocity") {
      auto v = get_input<zeno::NumericObject>("aux")->get<zeno::vec3f>();
      ls->getLevelSet() = typename ZenoLevelSet::uniform_vel_ls_t{
          zs::vec<float, 3>{v[0], v[1], v[2]}};
    } else
      throw std::runtime_error(
          fmt::format("unknown levelset (grid) category [{}].", cateStr));

    zs::match([](const auto &lsPtr) {
      if constexpr (zs::is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>) {
        using spls_t = typename RM_CVREF_T(lsPtr)::element_type;
        fmt::print(
            "levelset [{}] of dx [{}, {}], side_length [{}], block_size [{}]\n",
            spls_t::category, 1.f / lsPtr->_i2wSinv(0, 0), lsPtr->_grid.dx,
            spls_t::side_length, spls_t::block_size);
      } else if constexpr (zs::is_same_v<
                               typename RM_CVREF_T(lsPtr)::element_type,
                               typename ZenoLevelSet::uniform_vel_ls_t>) {
        fmt::print("uniform velocity field: {}, {}, {}\n", lsPtr->vel[0],
                   lsPtr->vel[1], lsPtr->vel[2]);
      } else {
        throw std::runtime_error(
            fmt::format("invalid levelset [{}] initialized in basicls.",
                        zs::get_var_type_str(lsPtr)));
      }
    })(ls->getBasicLevelSet()._ls);
    set_output("ZSLevelSet", std::move(ls));
  }
};
ZENDEFNODE(MakeZSLevelSet,
           {
               {{"float", "dx", "0.1"}, "aux"},
               {"ZSLevelSet"},
               {{"enum unknown apic flip aflip boundary", "transfer",
                 "unknown"},
                {"enum cellcentered collocated staggered const_velocity",
                 "category", "cellcentered"}},
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
    auto zsprim = get_input<ZenoParticles>("ZSParticles");
    auto &zspars = zsprim->getParticles();
    const auto size = zspars.size();

    auto prim = std::make_shared<PrimitiveObject>();
    prim->resize(size);

    using namespace zs;
    auto cudaExec = cuda_exec().device(0);

    static_assert(sizeof(zs::vec<float, 3>) == sizeof(zeno::vec3f),
                  "zeno::vec3f != zs::vec<float, 3>");
    /// verts
    for (auto &&prop : zspars.getPropertyTags()) {
      if (prop.numChannels == 3) {
        zs::Vector<zs::vec<float, 3>> dst{size, memsrc_e::device, 0};
        cudaExec(zs::range(size),
                 [zspars = zs::proxy<execspace_e::cuda>({}, zspars),
                  dst = zs::proxy<execspace_e::cuda>(dst),
                  name = prop.name] __device__(size_t pi) mutable {
                   // dst[pi] = zspars.pack<3>(name, pi);
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
/// elements
#if 1
    if (zsprim->isMeshPrimitive()) {
      auto &zseles = zsprim->getQuadraturePoints();
      int nVertsPerEle = static_cast<int>(zsprim->category) + 1;
      auto numEle = zseles.size();
      switch (zsprim->category) {
      case ZenoParticles::curve: {
        zs::Vector<zs::vec<int, 2>> dst{numEle, memsrc_e::device, 0};
        cudaExec(zs::range(numEle),
                 [zseles = zs::proxy<execspace_e::cuda>({}, zseles),
                  dst = zs::proxy<execspace_e::cuda>(
                      dst)] __device__(size_t ei) mutable {
                   dst[ei] = zseles.pack<2>("inds", ei).reinterpret_bits<int>();
                 });

        prim->lines.resize(numEle);
        auto &lines = prim->lines.values;
        copy(zs::mem_device, lines.data(), dst.data(),
             sizeof(zeno::vec2i) * numEle);
      } break;
      case ZenoParticles::surface: {
        zs::Vector<zs::vec<int, 3>> dst{numEle, memsrc_e::device, 0};
        cudaExec(zs::range(numEle),
                 [zseles = zs::proxy<execspace_e::cuda>({}, zseles),
                  dst = zs::proxy<execspace_e::cuda>(
                      dst)] __device__(size_t ei) mutable {
                   dst[ei] = zseles.pack<3>("inds", ei).reinterpret_bits<int>();
                 });

        prim->tris.resize(numEle);
        auto &tris = prim->tris.values;
        copy(zs::mem_device, tris.data(), dst.data(),
             sizeof(zeno::vec3i) * numEle);
      } break;
      case ZenoParticles::tet: {
        zs::Vector<zs::vec<int, 4>> dst{numEle, memsrc_e::device, 0};
        cudaExec(zs::range(numEle),
                 [zseles = zs::proxy<execspace_e::cuda>({}, zseles),
                  dst = zs::proxy<execspace_e::cuda>(
                      dst)] __device__(size_t ei) mutable {
                   dst[ei] = zseles.pack<4>("inds", ei).reinterpret_bits<int>();
                 });

        prim->quads.resize(numEle);
        auto &quads = prim->quads.values;
        copy(zs::mem_device, quads.data(), dst.data(),
             sizeof(zeno::vec4i) * numEle);
      } break;
      default:
        break;
      };
    }
#endif
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
