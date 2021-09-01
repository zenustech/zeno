#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimObject.h"
//#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/wrangler/Wrangler.hpp" //
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

// from projects/ZenoFX/pw.cpp : ParticlesWrangle
#include <cassert>
#include <cuda.h>
#include <zeno/DictObject.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/zeno.h>
#include <zensim/execution/ExecutionPolicy.hpp>
#include <zensim/physics/ConstitutiveModel.hpp>
#include <zfx/cuda.h>
#include <zfx/zfx.h>

namespace zeno {

struct ZSParticleNeighborWrangle : zeno::INode {
  struct Module {
    Module() = default;
    bool initialized() { return module != nullptr; }
    ~Module() {
      if (initialized())
        zs::cudri::unloadModuleData{module};
    }
    void *module{nullptr};
  } module;
  virtual void apply() override {
    using namespace zs;

    // parObjPtr
    auto parObjPtr = get_input<zeno::ZenoParticles>("ZSParticles");
    // parNeighborPtr
    std::shared_ptr<zeno::ZenoParticles> parNeighborPtr{};
    if (has_input<zeno::ZenoParticles>("ZSParticlesNeighbor"))
      parNeighborPtr = std::shared_ptr<zeno::ZenoParticles>(
          get_input<zeno::ZenoParticles>("ZSParticlesNeighbor").get(),
          [](void *) {});
    else {
      parNeighborPtr =
          std::make_shared<zeno::ZenoParticles>(*parObjPtr); // copy-ctor
      if constexpr (true) {
        fmt::print("indeed copied zeno particles!\n");
        auto &ref = parObjPtr->get();
        auto &neighbor = parNeighborPtr->get();
        match([](auto &&ref, auto &&neighbor) {
          fmt::print("particle obj comparison {} and {}!\n", demangle(ref),
                     demangle(neighbor));
          fmt::print("[size] ref {}, neighbor {}!\n", ref.size(),
                     neighbor.size());
          fmt::print("[memspace] ref {}, neighbor {}!\n",
                     get_memory_tag_name(ref.space()),
                     get_memory_tag_name(neighbor.space()));
          fmt::print("[devid] ref {}, neighbor {}!\n", (int)ref.devid(),
                     (int)neighbor.devid());
          fmt::print("[X addr] ref {}, neighbor {}!\n",
                     (void *)ref.attrVector("pos").data(),
                     (void *)neighbor.attrVector("pos").data());
        })(ref, neighbor);
      }
    }
    // ibs
    auto ibsPtr = get_input<zeno::ZenoIndexBuckets>("ZSIndexBuckets");
    // code
    auto code = get_input<zeno::StringObject>("zfxCode")->get();

    zfx::Compiler compiler;
    zfx::cuda::Assembler assembler;
    zfx::Options opts(zfx::Options::for_cuda);

    /// params
    auto params = has_input("params") ? get_input<zeno::DictObject>("params")
                                      : std::make_shared<zeno::DictObject>();
    std::vector<float> parvals;
    std::vector<std::pair<std::string, int>> parnames; // (paramName, dim)
    for (auto const &[key_, obj] : params->lut) {
      auto key = '$' + key_;
      auto par = zeno::smart_any_cast<std::shared_ptr<zeno::NumericObject>>(obj).get();
      auto dim = std::visit(
          [&](auto const &v) -> int {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, zeno::vec3f>) {
              parvals.push_back(v[0]);
              parvals.push_back(v[1]);
              parvals.push_back(v[2]);
              parnames.emplace_back(key, 0);
              parnames.emplace_back(key, 1);
              parnames.emplace_back(key, 2);
              return 3;
            } else if constexpr (std::is_same_v<T, float>) {
              parvals.push_back(v);
              parnames.emplace_back(key, 0);
              return 1;
            }
            return 0;
          },
          par->value);
      // fmt::print("define param: {} dim {}\n", key, dim);
      opts.define_param(key, dim);
    }

    /// symbols
    auto def_sym = [&opts](const std::string &key, int dim) {
      // fmt::print("define symbol: @{} dim {}\n", key, dim);
      opts.define_symbol('@' + key, dim);
      opts.define_symbol("@@" + key, dim);
    };

    match(
        [&](const auto &model, const auto &pars) {
          def_sym("mass", 1);
          def_sym("pos", pars.dim);
          def_sym("vel", pars.dim);
          def_sym("C", pars.dim * pars.dim);
          if constexpr (is_same_v<RM_CVREF_T(model), FixedCorotatedConfig>)
            def_sym("F", pars.dim * pars.dim);
          else if constexpr (is_same_v<RM_CVREF_T(model),
                                       EquationOfStateConfig>)
            def_sym("J", pars.dim * pars.dim);
        },
        [](...) {})(parObjPtr->model, parObjPtr->get());

    auto prog = compiler.compile(code, opts);
    auto jitCode = assembler.assemble(prog->assembly);

    auto &currentContext = Cuda::context(0);
    currentContext.setContext();
    if (!module.initialized()) {
      /// execute on the current particle object
      auto wrangleKernelPtxs = cudri::load_all_ptx_files_at();
      void *state;
      cudri::linkCreate(0, nullptr, nullptr, &state);

      auto jitSrc = cudri::compile_cuda_source_to_ptx(jitCode);
      cudri::linkAddData(state, CU_JIT_INPUT_PTX, (void *)jitSrc.data(),
                         (size_t)jitSrc.size(), "script", 0, NULL, NULL);

      int no = 0;
      for (auto const &ptx : wrangleKernelPtxs) {
        auto str = std::string("wrangler") + std::to_string(no++);
        cudri::linkAddData(state, CU_JIT_INPUT_PTX, (char *)ptx.data(),
                           ptx.size(), str.data(), 0, NULL, NULL);
      }
      void *cubin;
      size_t cubinSize;
      cudri::linkComplete(state, &cubin, &cubinSize);

      cudri::loadModuleData(&module.module, cubin);
      cudri::linkDestroy{state};
    }

    /// symbols
    zs::Vector<AccessorAoSoA> haccessors{prog->symbols.size()};
    auto unitBytes = match([](auto &pars) {
      return sizeof(typename RM_CVREF_T(pars)::T);
    })(parObjPtr->get());
    const int dim = match([](auto &pars) { return RM_CVREF_T(pars)::dim; })(
        parObjPtr->get());
    for (int i = 0; i < prog->symbols.size(); i++) {
      auto [name, dimid] = prog->symbols[i];
      // printf("channel %d: %s.%d\t", i, name.c_str(), dimid);
      bool isNeighborChannel = false;
      auto targetParPtr = &parObjPtr->get(); // GeneralParticles
      if (name[1] == '@') {
        isNeighborChannel = true;
        name = name.substr(2);
        targetParPtr = &parNeighborPtr->get();
      } else
        name = name.substr(1);

      int ndim = 1;
      void *addr = nullptr;
      match([&ndim, &addr, dim, name = name](auto &pars) {
        if (name == "mass") {
          ndim = 1;
          addr = pars.attrScalar("mass").data();
        } else if (name == "pos") {
          ndim = dim;
          addr = pars.attrVector("pos").data();
        } else if (name == "vel") {
          ndim = dim;
          addr = pars.attrVector("vel").data();
        } else if (name == "C") {
          ndim = dim * dim;
          addr = pars.attrMatrix("C").data();
        } else if (name == "F") {
          ndim = dim * dim;
          addr = pars.attrMatrix("F").data();
        } else if (name == "J") {
          ndim = 1;
          addr = pars.attrScalar("J").data();
        } else if (name == "logJp") {
          ndim = 1;
          addr = pars.attrScalar("logJp").data();
        }
      })(*targetParPtr);
      haccessors[i] = zs::AccessorAoSoA{zs::aos_v,
                                        addr,
                                        (unsigned short)unitBytes,
                                        (unsigned short)ndim,
                                        (unsigned short)dimid,
                                        (unsigned short)isNeighborChannel};
      // fmt::print("base: {}\n", haccessors[i].base);
    }
    auto daccessors = haccessors.clone({zs::memsrc_e::device, 0});

    /// params
    zs::Vector<zs::f32> hparams{prog->params.size()};
    for (int i = 0; i < prog->params.size(); i++) {
      auto [name, dimid] = prog->params[i];
      // printf("parameter %d: %s.%d\t", i, name.c_str(), dimid);
      auto it = std::find(parnames.begin(), parnames.end(),
                          std::make_pair(name, dimid));
      auto value = parvals.at(it - parnames.begin());
      // printf("(valued %f)\n", value);
      hparams[i] = value;
    }
    zs::Vector<zs::f32> dparams = hparams.clone({zs::memsrc_e::device, 0});

    void *function;
    cudri::getModuleFunc(&function, module.module,
                         "zpc_particle_neighbor_wrangle_kernel");

    // begin kernel launch
    std::size_t cnt;
    zs::f32 *d_params;
    zs::VectorView<zs::execspace_e::cuda, zs::Vector<zs::vec<zs::f32, 3>>>
        posView, neighborPosView;
    zs::IndexBucketsView<zs::execspace_e::cuda, zs::IndexBuckets<3>> ibsView =
        proxy<zs::execspace_e::cuda>(
            std::get<zs::IndexBuckets<3>>(ibsPtr->get()));
    zs::ParticlesView<zs::execspace_e::cuda, zs::Particles<zs::f32, 3>> parObj;
    int nchns = daccessors.size();
    void *addr = daccessors.data();
    void *args[7];

    match(
        [&](auto &pars, auto &neighbor)
            -> std::enable_if_t<
                zs::is_same_v<RM_CVREF_T(pars), zs::Particles<zs::f32, 3>> &&
                zs::is_same_v<RM_CVREF_T(neighbor),
                              zs::Particles<zs::f32, 3>>> {
          cnt = pars.size();
          args[0] = (void *)&cnt;
          posView = proxy<zs::execspace_e::cuda>(pars.attrVector("pos"));
          args[1] = (void *)&posView;
          neighborPosView =
              proxy<zs::execspace_e::cuda>(neighbor.attrVector("pos"));
          args[2] = (void *)&neighborPosView;

          args[3] = (void *)&ibsView;
          d_params = dparams.data();
          args[4] = (void *)&d_params;
          args[5] = (void *)&nchns;
          args[6] = (void *)&addr;
        },
        [](...) {})(parObjPtr->get(), parNeighborPtr->get());
    cudri::launchCuKernel(function, (cnt + 127) / 128, 1, 1, 128, 1, 1, 0,
                          currentContext.streamSpare(0), args,
                          (void **)nullptr);
    // end kernel launch

    cudri::syncContext();

    parNeighborPtr.reset();
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ZSParticleNeighborWrangle,
           {
               {"ZSParticles", "ZSParticlesNeighbor", "ZSIndexBuckets",
                "zfxCode", "params"},
               {"ZSParticles"},
               {},
               {"GPUMPM"},
           });

} // namespace zeno
