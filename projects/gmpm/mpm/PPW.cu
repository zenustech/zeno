#include "../Utils.hpp"
#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/wrangler/Wrangler.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

// from projects/ZenoFX/ppw.cpp : ParticlesParticlesWrangle
#include <cassert>
#include <cuda.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
#include <zensim/execution/ExecutionPolicy.hpp>
#include <zensim/physics/ConstitutiveModel.hpp>
#include <zfx/cuda.h>
#include <zfx/zfx.h>

namespace zeno {

static zfx::Compiler compiler;
static zfx::cuda::Assembler assembler;

struct ZSParticleParticleWrangler : INode {
  ~ZSParticleParticleWrangler() {
    if (this->_cuModule)
      zs::cudri::unloadModuleData(this->_cuModule);
  }
  void apply() override {
    using namespace zs;
    auto code = get_input<StringObject>("zfxCode")->get();

    /// parObjPtr
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    if (parObjPtrs.size() > 1)
      throw std::runtime_error(
          "zs ppw currently only supports up to one particle object.");
    auto parObjPtr = parObjPtrs[0];
    auto &pars = parObjPtr->getParticles();
    auto props = pars.getPropertyTags();
    // auto parObjPtr = get_input<ZenoParticles>("ZSParticles");

    /// parNeighborPtr
    std::shared_ptr<ZenoParticles> parNeighborPtr{};
    if (has_input<ZenoParticles>("ZSNeighborParticles"))
      parNeighborPtr = get_input<ZenoParticles>("ZSNeighborParticles");
    else if (!has_input("ZSNeighborParticles"))
      parNeighborPtr = std::make_shared<ZenoParticles>(*parObjPtr); // copy-ctor
    else
      throw std::runtime_error(
          "something strange passed to zs pnw as the neighbor particles.");
    const auto &neighborPars = parNeighborPtr->getParticles();
    const auto neighborProps = neighborPars.getPropertyTags();

    zfx::Options opts(zfx::Options::for_cuda);
    opts.detect_new_symbols = true;

    /// params
    auto params = has_input("params") ? get_input<zeno::DictObject>("params")
                                      : std::make_shared<zeno::DictObject>();
    std::vector<float> parvals;
    std::vector<std::pair<std::string, int>> parnames; // (paramName, dim)
    for (auto const &[key_, obj] : params->lut) {
      auto key = '$' + key_;
      if (auto o = zeno::silent_any_cast<zeno::NumericValue>(obj);
          o.has_value()) {
        auto par = o.value();
        auto dim = std::visit(
            [&](auto const &v) {
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
              } else {
                printf("invalid parameter type encountered: `%s`\n",
                       typeid(T).name());
                return 0;
              }
            },
            par);
        opts.define_param(key, dim);
      }
    }

    /// symbols
    auto def_sym = [&opts](const std::string &prefix, const std::string &key,
                           int dim) { opts.define_symbol(prefix + key, dim); };

    opts.symdims.clear();
    for (auto &&[name, nchns] : props)
      def_sym("@", name.asString(), nchns);
    for (auto &&[name, nchns] : neighborProps)
      def_sym("@@", name.asString(), nchns);

    auto &currentContext = Cuda::context(0);
    currentContext.setContext();
    auto cudaPol = cuda_exec().device(0).sync(true);

    auto prog = compiler.compile(code, opts);
    auto jitCode = assembler.assemble(prog->assembly);

    /// supplement new properties
    auto checkDuplication = [](std::string_view tag,
                               const auto &props) -> bool {
      for (auto &&[name, nchns] : props)
        if (name == tag)
          return true;
      return false;
    };
    // adding channels is for particles only!
    std::vector<zs::PropertyTag> newChns{};
    for (auto const &[name, dim] : prog->newsyms) {
      assert(name[0] == '@');
      if (name.substr(0, 2) == "@@") { // channel is from neighbor
        auto key = name.substr(2);
        if (!checkDuplication(key, neighborProps)) {
          auto msg = fmt::format(
              "property [{}] is not present in the neighbor particles.", key);
          throw std::runtime_error(msg);
        }
      } else {
        auto key = name.substr(1);
        if (!checkDuplication(key, props))
          newChns.push_back(PropertyTag{key, dim});
      }
    }
    if (newChns.size() > 0)
      pars.append_channels(cudaPol, newChns);
    props.insert(std::end(props), std::begin(newChns), std::end(newChns));

    if (_cuModule == nullptr) {
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

      cudri::loadModuleData(&_cuModule, cubin);
      cudri::linkDestroy{state};
    }

    /// symbols
    zs::Vector<AccessorAoSoA> haccessors{prog->symbols.size()};
    auto unitBytes = sizeof(RM_CVREF_T(pars)::value_type);
    constexpr auto tileSize = RM_CVREF_T(pars)::lane_width;

    for (int i = 0; i < prog->symbols.size(); i++) {
      auto [name, dimid] = prog->symbols[i];
      bool isNeighborProperty = false;
      auto targetParPtr = &neighborPars;
      if (name.substr(0, 2) == "@@") {
        isNeighborProperty = true;
        name = name.substr(2);
      } else {
        targetParPtr = &pars;
        name = name.substr(1);
      }

      haccessors[i] = zs::AccessorAoSoA{
          zs::aosoa_c,
          (void *)targetParPtr->data(),
          (unsigned short)unitBytes,
          (unsigned short)tileSize,
          (unsigned short)targetParPtr->numChannels(),
          (unsigned short)(targetParPtr->getChannelOffset(name) + dimid),
          (unsigned short)isNeighborProperty};
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
      hparams[i] = value;
    }
    zs::Vector<zs::f32> dparams = hparams.clone({zs::memsrc_e::device, 0});

    void *function;
    cudri::getModuleFunc(&function, _cuModule,
                         "zpc_particle_particle_wrangler_kernel");

    // begin kernel launch
    std::size_t cnt = pars.size();
    std::size_t cntNei = neighborPars.size();
    zs::f32 *d_params = dparams.data();
    int nchns = daccessors.size();
    void *addr = daccessors.data();
    void *args[] = {(void *)&cnt, (void *)&cntNei, (void *)&d_params,
                    (void *)&nchns, (void *)&addr};

    cudri::launchCuKernel(function, (cnt + 127) / 128, 1, 1, 128, 1, 1, 0,
                          currentContext.streamSpare(0), args,
                          (void **)nullptr);
    // end kernel launch
    cudri::syncContext();

    set_output("ZSParticles", get_input("ZSParticles"));
  }

private:
  void *_cuModule{nullptr};
};

ZENDEFNODE(ZSParticleParticleWrangler,
           {
               {{"ZenoParticles", "ZSParticles"},
                {"ZenoParticles", "ZSNeighborParticles"},
                {"string", "zfxCode"},
                {"DictObject:NumericObject", "params"}},
               {"ZSParticles"},
               {},
               {"MPM"},
           });

} // namespace zeno
