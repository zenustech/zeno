#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/wrangler/Wrangler.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

// from projects/ZenoFX/pnw.cpp : ParticlesNeighborWrangle
#include "dbg_printf.h"
#include <cassert>
#include <cuda.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <zensim/execution/ExecutionPolicy.hpp>
#include <zensim/physics/ConstitutiveModel.hpp>
#include <zfx/cuda.h>
#include <zfx/zfx.h>

namespace zeno {

static zfx::Compiler compiler;
static zfx::cuda::Assembler assembler;

struct ZSParticleNeighborWrangler : INode {
    ~ZSParticleNeighborWrangler() {
        if (this->_cuModule)
            cuModuleUnload((CUmodule)this->_cuModule);
    }
    void apply() override {
        using namespace zs;

        auto &currentContext = Cuda::context(0);
        currentContext.setContext();
        auto cudaPol = cuda_exec().device(0).sync(true);

        auto code = get_input<StringObject>("zfxCode")->get();

        /// parObjPtr
        auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        if (parObjPtrs.size() > 1)
            throw std::runtime_error("zs pnw currently only supports up to one particle object.");
        auto parObjPtr = parObjPtrs[0];
        auto &pars = parObjPtr->getParticles();
        auto props = pars.getPropertyTags();
        // auto parObjPtr = get_input<ZenoParticles>("ZSParticles");

        /// parNeighborPtr
        auto neighborParObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSNeighborParticles");
        std::shared_ptr<ZenoParticles> parNeighborPtr{};
        if (neighborParObjPtrs.size() > 0)
            parNeighborPtr = std::shared_ptr<ZenoParticles>(neighborParObjPtrs[0], [](void *) {});
        else if (!has_input("ZSNeighborParticles"))
            parNeighborPtr = std::make_shared<ZenoParticles>(*parObjPtr); // copy-ctor
        else
            throw std::runtime_error("something strange passed to zs pnw as the neighbor particles.");
        const auto &neighborPars = parNeighborPtr->getParticles();
        const auto neighborProps = neighborPars.getPropertyTags();

        /// ibs (TODO: generate based on neighborPars, when this input is absent)
        std::shared_ptr<ZenoIndexBuckets> ibsPtr{};
        if (has_input<ZenoIndexBuckets>("ZSIndexBuckets"))
            ibsPtr = get_input<ZenoIndexBuckets>("ZSIndexBuckets");
        else if (has_input<NumericObject>("ZSIndexBuckets"))
            spatial_hashing(cudaPol, neighborPars, get_input<NumericObject>("ZSIndexBuckets")->get<float>() * 2,
                            ibsPtr->get());
        else
            ;
        const auto &ibs = ibsPtr->get();

        zfx::Options opts(zfx::Options::for_cuda);
        opts.detect_new_symbols = true;

        /// params
        auto params =
            has_input("params") ? get_input<zeno::DictObject>("params") : std::make_shared<zeno::DictObject>();
        {
            // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
            auto const &gs = *this->getGlobalState();
            params->lut["PI"] = objectFromLiterial((float)(std::atan(1.f) * 4));
            params->lut["F"] = objectFromLiterial((float)gs.frameid);
            params->lut["DT"] = objectFromLiterial(gs.frame_time);
            params->lut["T"] = objectFromLiterial(gs.frame_time * gs.frameid + gs.frame_time_elapsed);
            // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
            // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
            for (auto const &[key, ref] : getThisGraph()->portalIns) {
                if (auto i = code.find('$' + key); i != std::string::npos) {
                    i = i + key.size() + 1;
                    if (code.size() <= i || !std::isalnum(code[i])) {
                        if (params->lut.count(key))
                            continue;
                        dbg_printf("ref portal %s\n", key.c_str());
                        auto res =
                            getThisGraph()->callTempNode("PortalOut", {{"name:", objectFromLiterial(key)}}).at("port");
                        params->lut[key] = std::move(res);
                    }
                }
            }
            // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
            // BEGIN伺候心欣伺候懒得extract出变量了
            std::vector<std::string> keys;
            for (auto const &[key, val] : params->lut) {
                keys.push_back(key);
            }
            for (auto const &key : keys) {
                if (!dynamic_cast<zeno::NumericObject *>(params->lut.at(key).get())) {
                    dbg_printf("ignored non-numeric %s\n", key.c_str());
                    params->lut.erase(key);
                }
            }
            // END伺候心欣伺候懒得extract出变量了
        }
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames; // (paramName, dim)
        for (auto const &[key_, par] : params->getLiterial<zeno::NumericValue>()) {
            auto key = '$' + key_;
            auto dim = std::visit(
                [&](auto const &v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_convertible_v<T, zeno::vec3f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        parnames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr (std::is_convertible_v<T, float>) {
                        parvals.push_back(v);
                        parnames.emplace_back(key, 0);
                        return 1;
                    } else {
                        printf("invalid parameter type encountered: `%s`\n", typeid(T).name());
                        return 0;
                    }
                },
                par);
            //dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
            opts.define_param(key, dim);
            //auto par = zeno::safe_any_cast<zeno::NumericValue>(obj);
        }

        /// symbols
        auto def_sym = [&opts](const std::string &prefix, std::string key, int dim) {
            if (key == "x")
                opts.define_symbol(prefix + "pos", dim);
            else if (key == "v")
                opts.define_symbol(prefix + "vel", dim);
            opts.define_symbol(prefix + key, dim);
        };

        opts.symdims.clear();
        for (auto &&[name, nchns] : props)
            def_sym("@", name.asString(), nchns);
        for (auto &&[name, nchns] : neighborProps)
            def_sym("@@", name.asString(), nchns);

        auto prog = compiler.compile(code, opts);
        auto jitCode = assembler.assemble(prog->assembly);

        /// supplement new properties
        auto checkDuplication = [](std::string_view tag, const auto &props) -> bool {
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
                    auto msg = fmt::format("property [{}] is not present in the neighbor particles.", key);
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

        if (_cuModule == nullptr) {
            /// execute on the current particle object
            auto wrangleKernelPtxs = cudri::load_all_ptx_files_at();
            void *state;
            cuLinkCreate(0, nullptr, nullptr, (CUlinkState *)&state);

            auto jitSrc = cudri::compile_cuda_source_to_ptx(jitCode);
            cuLinkAddData((CUlinkState)state, CU_JIT_INPUT_PTX, (void *)jitSrc.data(), (size_t)jitSrc.size(), "script",
                          0, NULL, NULL);

            int no = 0;
            for (auto const &ptx : wrangleKernelPtxs) {
                auto str = std::string("wrangler") + std::to_string(no++);
                cuLinkAddData((CUlinkState)state, CU_JIT_INPUT_PTX, (char *)ptx.data(), ptx.size(), str.data(), 0, NULL,
                              NULL);
            }
            void *cubin;
            size_t cubinSize;
            cuLinkComplete((CUlinkState)state, &cubin, &cubinSize);

            cuModuleLoadData((CUmodule *)&_cuModule, cubin);
            cuLinkDestroy((CUlinkState)state);
        }

        auto transTag = [](std::string str) {
            if (str == "pos")
                str = "x";
            else if (str == "vel")
                str = "v";
            return str;
        };

        /// symbols
        zs::Vector<AccessorAoSoA> haccessors{prog->symbols.size()};
        auto unitBytes = sizeof(RM_CVREF_T(pars)::value_type);
        constexpr auto tileSize = RM_CVREF_T(pars)::lane_width;
#if 0
    static_assert(tileSize == RM_CVREF_T(neighborPars)::lane_width,
                  "target particles object and neighboring particles are of "
                  "different layout!");
#endif

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

            haccessors[i] = zs::AccessorAoSoA{zs::aosoa_c,
                                              (void *)targetParPtr->data(),
                                              (unsigned short)unitBytes,
                                              (unsigned short)tileSize,
                                              (unsigned short)targetParPtr->numChannels(),
                                              (unsigned short)(targetParPtr->getPropertyOffset(transTag(name)) + dimid),
                                              (unsigned short)isNeighborProperty};
        }
        auto daccessors = haccessors.clone({zs::memsrc_e::device, 0});

        /// params
        zs::Vector<zs::f32> hparams{prog->params.size()};
        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            // printf("parameter %d: %s.%d\t", i, name.c_str(), dimid);
            auto it = std::find(parnames.begin(), parnames.end(), std::make_pair(name, dimid));
            auto value = parvals.at(it - parnames.begin());
            hparams[i] = value;
        }
        zs::Vector<zs::f32> dparams = hparams.clone({zs::memsrc_e::device, 0});

        void *function;
        cuModuleGetFunction((CUfunction *)&function, (CUmodule)_cuModule, "zpc_particle_neighbor_wrangler_kernel");

        // begin kernel launch
        std::size_t cnt = pars.size();
        auto parsv = zs::proxy<zs::execspace_e::cuda>({}, pars);
        auto neighborParsv = zs::proxy<zs::execspace_e::cuda>({}, neighborPars);
        auto ibsv = zs::proxy<zs::execspace_e::cuda>(ibs);
        zs::f32 *d_params = dparams.data();
        int nchns = daccessors.size();
        void *addr = daccessors.data();
        int isBox = get_input2<bool>("is_box") ? 1 : 0;
        float radius = ibs._dx;
        void *args[] = {(void *)&cnt,  (void *)&isBox,    (void *)&radius, (void *)&parsv, (void *)&neighborParsv,
                        (void *)&ibsv, (void *)&d_params, (void *)&nchns,  (void *)&addr};

        cuLaunchKernel((CUfunction)function, (cnt + 127) / 128, 1, 1, 128, 1, 1, 0,
                       (CUstream)currentContext.streamSpare(-1), args, (void **)nullptr);
        // end kernel launch
        cuCtxSynchronize();

        set_output("ZSParticles", get_input("ZSParticles"));
    }

  private:
    void *_cuModule{nullptr};
};

ZENDEFNODE(ZSParticleNeighborWrangler, {
                                           {{"ZenoParticles", "ZSParticles"},
                                            {"ZenoParticles", "ZSNeighborParticles"},
                                            {"ZenoIndexBuckets", "ZSIndexBuckets"},
                                            {"string", "zfxCode"},
                                            {"bool", "is_box", "1"},
                                            {"DictObject:NumericObject", "params"}},
                                           {"ZSParticles"},
                                           {},
                                           {"zswrangle"},
                                       });

} // namespace zeno
