#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/wrangler/Wrangler.hpp" //
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

// from projects/ZenoFX/p2w.cpp : ParticlesTwoWrangle
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
#include <zfx/cuda.h>
#include <zfx/zfx.h>

namespace zeno {

static zfx::Compiler compiler;
static zfx::cuda::Assembler assembler;

struct ZSParticlesTwoWrangler : zeno::INode {
    ~ZSParticlesTwoWrangler() {
        if (this->_cuModule)
            cuModuleUnload((CUmodule)this->_cuModule);
    }
    void apply() override {
        using namespace zs;
        auto code = get_input<StringObject>("zfxCode")->get();

        auto parObjPtr = get_input<ZenoParticles>("ZSParticles");
        auto parObjPtr2 = get_input<ZenoParticles>("ZSParticles2");

        if (parObjPtr->numParticles() != parObjPtr2->numParticles()) {
            dbg_printf("prim and prim2 size mismatch (%d != %d), using minimal\n", parObjPtr->numParticles(),
                       parObjPtr2->numParticles());
        }

        zfx::Options opts(zfx::Options::for_cuda);
        opts.detect_new_symbols = true;
        // opts.reassign_channels = true;
        // opts.reassign_parameters = false;

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
        std::vector<std::pair<std::string, int>> parnames;
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
                    } else if constexpr (std::is_convertible_v<T, vec2f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        return 2;
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
        }

        auto &currentContext = Cuda::context(0);
        currentContext.setContext();
        auto cudaPol = cuda_exec().sync(true);

        /// symbols
        {
            auto &pars = parObjPtr->getParticles();
            auto &pars2 = parObjPtr2->getParticles();
            auto props = pars.getPropertyTags();
            auto props2 = pars2.getPropertyTags();
            opts.symdims.clear();
            // PropertyTag can be used for structured binding automatically
            auto register_attrib_syms = [&opts](const std::vector<PropertyTag> &props, const std::string &prefix) {
                for (auto [name, nchns] : props) {
                    if (name == "x")
                        opts.define_symbol(prefix + "pos", nchns);
                    else if (name == "v")
                        opts.define_symbol(prefix + "vel", nchns);
                    opts.define_symbol(prefix + std::string(name), nchns);
                }
                //def_sym(std::string(name), nchns);
            };
            register_attrib_syms(props, "@");
            register_attrib_syms(props2, "@@");

            auto prog = compiler.compile(code, opts);
            auto jitCode = assembler.assemble(prog->assembly);

            /// supplement new properties
            auto checkDuplication = [&props](std::string_view tag) -> bool {
                for (auto &&[name, nchns] : props)
                    if (name == tag.data())
                        return true;
                return false;
            };
            std::vector<zs::PropertyTag> newChns{};
            for (auto const &[name, dim] : prog->newsyms) {
                assert(name[0] == '@');
                if (name[1] == '@')
                    err_printf("ERROR: cannot define new attribute %s on prim2\n", name.c_str());
                auto key = name.substr(1);
                if (!checkDuplication(key))
                    newChns.push_back(PropertyTag{key, dim});
                if (dim != 3 && dim != 1)
                    err_printf("ERROR: bad attribute dimension for primitive: %d\n", dim);
            }
            if (newChns.size() > 0)
                pars.append_channels(cudaPol, newChns);

            if (_cuModule == nullptr) {
                auto wrangleKernelPtxs = cudri::load_all_ptx_files_at();
                void *state;
                cuLinkCreate(0, nullptr, nullptr, (CUlinkState *)&state);
                // fmt::print(">>>>>>>>>>>>>>>>>>>");
                // fmt::print("{}\n", jitCode);
                // fmt::print("<<<<<<<<<<<<<<<<<<<");

                auto jitSrc = cudri::compile_cuda_source_to_ptx(jitCode);
                cuLinkAddData((CUlinkState)state, CU_JIT_INPUT_PTX, (void *)jitSrc.data(), (size_t)jitSrc.size(),
                              "script", 0, NULL, NULL);

                int no = 0;
                for (auto const &ptx : wrangleKernelPtxs) {
                    auto str = std::string("wrangler") + std::to_string(no++);
                    cuLinkAddData((CUlinkState)state, CU_JIT_INPUT_PTX, (char *)ptx.data(), ptx.size(), str.data(), 0,
                                  NULL, NULL);
                }
                void *cubin;
                size_t cubinSize;
                cuLinkComplete((CUlinkState)state, &cubin, &cubinSize);

                cuModuleLoadData((CUmodule *)&_cuModule, cubin);
                cuLinkDestroy((CUlinkState)state);
            }

            zs::Vector<AccessorAoSoA> haccessors{prog->symbols.size()};
            auto unitBytes = sizeof(RM_CVREF_T(pars)::value_type);
            constexpr auto tileSize = RM_CVREF_T(pars)::lane_width;

            auto transTag = [](std::string str) {
                if (str == "pos")
                    str = "x";
                else if (str == "vel")
                    str = "v";
                return str;
            };

            /// symbols
            for (int i = 0; i < prog->symbols.size(); i++) {
                auto [name, dimid] = prog->symbols[i];

                unsigned short aux = name[1] == '@' ? 1 : 0;
                float *parsPtr = nullptr;
                if (aux) {
                    aux = 1;
                    parsPtr = pars2.data();
                    name = name.substr(2);
                } else {
                    parsPtr = pars.data();
                    name = name.substr(1);
                }
                const auto &curPars = aux ? pars2 : pars;
                haccessors[i] = zs::AccessorAoSoA{zs::aosoa_c,
                                                  parsPtr,
                                                  (unsigned short)unitBytes,
                                                  (unsigned short)tileSize,
                                                  (unsigned short)curPars.numChannels(),
                                                  (unsigned short)(curPars.getPropertyOffset(transTag(name)) + dimid),
                                                  aux};
#if 0
                fmt::print("channel {}: {}.{}. chn offset: {} (of {})\n", i, name.c_str(), dimid,
                           curPars.getPropertyOffset(name), curPars.numChannels());
#endif
            }
            auto daccessors = haccessors.clone({zs::memsrc_e::device});

            /// params
            zs::Vector<zs::f32> hparams{prog->params.size()};
            for (int i = 0; i < prog->params.size(); i++) {
                auto [name, dimid] = prog->params[i];
                // printf("parameter %d: %s.%d\t", i, name.c_str(), dimid);
                auto it = std::find(parnames.begin(), parnames.end(), std::make_pair(name, dimid));
                auto value = parvals.at(it - parnames.begin());
                hparams[i] = value;
            }
            zs::Vector<zs::f32> dparams = hparams.clone({zs::memsrc_e::device});

            void *function;
            cuModuleGetFunction((CUfunction *)&function, (CUmodule)_cuModule, "zpc_particle_wrangler_kernel");

            // begin kernel launch
            std::size_t cnt = pars.size();
            zs::f32 *d_params = dparams.data();
            int nchns = daccessors.size();
            void *addr = daccessors.data();
            void *args[4] = {(void *)&cnt, (void *)&d_params, (void *)&nchns, (void *)&addr};

            cuLaunchKernel((CUfunction)function, (cnt + 127) / 128, 1, 1, 128, 1, 1, 0,
                           (CUstream)currentContext.streamSpare(-1), args, (void **)nullptr);
            // end kernel launch
            cuCtxSynchronize();
        }

        set_output("ZSParticles", get_input("ZSParticles"));
    }

  private:
    void *_cuModule{nullptr};
};

ZENDEFNODE(ZSParticlesTwoWrangler, {
                                       {{"ZenoParticles", "ZSParticles"},
                                        {"ZenoParticles", "ZSParticles2"},
                                        {"string", "zfxCode"},
                                        {"DictObject:NumericObject", "params"}},
                                       {"ZSParticles"},
                                       {},
                                       {"zswrangle"},
                                   });

} // namespace zeno
