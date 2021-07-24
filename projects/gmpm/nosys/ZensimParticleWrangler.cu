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

struct ZSParticlesWrangle : zeno::INode {
  virtual void apply() override {
    using namespace zs;
    auto code = get_input<zeno::StringObject>("zfxCode")->get();

#if 1
    zeno::ZenoParticleObjects parObjPtrs{};
    if (get_input("ZSParticles")->as<zeno::ZenoParticles>())
      parObjPtrs.push_back(get_input("ZSParticles")->as<zeno::ZenoParticles>());
    else if (get_input("ZSParticles")->as<zeno::ZenoParticleList>()) {
      auto &list =
          get_input("ZSParticles")->as<zeno::ZenoParticleList>()->get();
      parObjPtrs.insert(parObjPtrs.end(), list.begin(), list.end());
    } else if (get_input("ZSParticles")->as<zeno::ListObject>()) {
      auto &objSharedPtrLists =
          *get_input("ZSParticles")->as<zeno::ListObject>();
      for (auto &&objSharedPtr : objSharedPtrLists.arr)
        if (auto ptr = dynamic_cast<zeno::ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }

    zfx::Compiler compiler;
    zfx::cuda::Assembler assembler;
    zfx::Options opts(zfx::Options::for_cuda);
    opts.reassign_channels = false;
    opts.reassign_parameters = false;

    /// params
    auto params = has_input("params") ? get_input<zeno::DictObject>("params")
                                      : std::make_shared<zeno::DictObject>();
    std::vector<float> parvals;
    std::vector<std::pair<std::string, int>> parnames; // (paramName, dim)
    zs::Vector<zs::f32> dparams;
    for (auto const &[key_, obj] : params->lut) {
      auto key = '$' + key_;
      auto par = dynamic_cast<zeno::NumericObject *>(obj.get());
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
      fmt::print("define param: {} dim {}\n", key, dim);
      opts.define_param(key, dim);
    }
    zs::Vector<zs::f32> hparams{parvals.size()};
    for (auto &&[dst, src] : zip(hparams, parvals))
      dst = src;
    dparams = hparams.clone({zs::memsrc_e::device, 0});
    /// symbols
    auto def_sym = [&opts](const std::string &key, int dim) {
      fmt::print("define symbol: @{} dim {}\n", key, dim);
      opts.define_symbol('@' + key, dim);
    };
    for (auto &&parObjPtr : parObjPtrs) {
      fmt::print("\n[defining symbols] iterating zsparticles [{}]: \n",
                 (void *)parObjPtr);
      opts.symdims.clear();
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
      auto jitCode = assembler.assemble(
          prog->assembly); // amazing! you avoid nvrtc totally

      for (int i = 0; i < prog->symbols.size(); i++) {
        auto [name, dimid] = prog->symbols[i];
        printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
      }

      if constexpr (true) { /// execute on the current particle object
        auto wrangleKernelPtxs = cudri::load_all_ptx_files_at();
        void *state;
        cudri::linkCreate(0, nullptr, nullptr, &state);

#if 1
        auto jitSrc = cudri::compile_cuda_source_to_ptx(jitCode);
        cudri::linkAddData(state, CU_JIT_INPUT_PTX, (void *)jitSrc.data(),
                           (size_t)jitSrc.size(), "script", 0, NULL, NULL);
#else
        cudri::linkAddData(state, CU_JIT_INPUT_PTX, (void *)jitCode.data(),
                           (size_t)jitCode.size(), "script", 0, NULL, NULL);
#endif

        int no = 0;
        for (auto const &ptx : wrangleKernelPtxs) {
          auto str = std::string("wrangler") + std::to_string(no++);
          cudri::linkAddData(state, CU_JIT_INPUT_PTX, (char *)ptx.data(),
                             ptx.size(), str.data(), 0, NULL, NULL);
        }
        void *cubin;
        size_t cubinSize;
        cudri::linkComplete(state, &cubin, &cubinSize);

        void *module;
        cudri::loadModuleData(&module, cubin);

        void *function;
        cudri::getModuleFunc(&function, module, "zpc_particle_wrangle_kernel");

        auto &currentContext = Cuda::context(0);

        // begin kernel launch
        std::size_t cnt;
        zs::f32 *d_params;
        zs::ParticlesProxy<zs::execspace_e::cuda, zs::Particles<zs::f32, 3>>
            parObj;
        void *args[4];

        match(
            [&](auto &pars) -> std::enable_if_t<std::is_same_v<
                                RM_CVREF_T(pars), zs::Particles<zs::f32, 3>>> {
              cnt = pars.size();
              args[0] = (void *)&cnt;
              d_params = dparams.data();
              args[1] = (void *)&d_params;
              parObj = proxy<zs::execspace_e::cuda>(pars);
              args[2] = (void *)&parObj;
              args[3] = (void *)&parObjPtr->model;
            },
            [](...) {})(parObjPtr->get());
        cudri::launchCuKernel(function, (cnt + 127) / 128, 1, 1, 128, 1, 1, 0,
                              currentContext.streamSpare(0), args,
                              (void **)nullptr);
        // end kernel launch

        cudri::syncContext();

        cudri::unloadModuleData{module};
        cudri::linkDestroy{state};
      }
    }
#endif

    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ZSParticlesWrangle, {
                                   {"ZSParticles", "zfxCode", "params"},
                                   {"ZSParticles"},
                                   {},
                                   {"GPUMPM"},
                               });

} // namespace zeno