#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/wrangler/Wrangler.hpp" //
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

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

struct ZSVolumeWrangler : zeno::INode {
    ~ZSVolumeWrangler() {
        if (this->_cuModule)
            cuModuleUnload((CUmodule)this->_cuModule);
    }
    void apply() override {
        using namespace zs;
        auto code = get_input<StringObject>("zfxCode")->get();

        auto spgPtrs = RETRIEVE_OBJECT_PTRS(ZenoSparseGrid, "ZSGrid");

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
            //auto par = zeno::safe_any_cast<zeno::NumericValue>(obj);
        }

        auto &currentContext = Cuda::context(0);
        currentContext.setContext();
        auto cudaPol = cuda_exec().device(0).sync(true);

        /// symbols
        auto def_sym = [&opts](std::string key, int dim, bool isDoubleBuffer) {
            if (isDoubleBuffer)
                key.pop_back();

            if (key == "x")
                opts.define_symbol("@pos", dim);
            else if (key == "v")
                opts.define_symbol("@vel", dim);
            opts.define_symbol('@' + key, dim);
            if (isDoubleBuffer) {
                if (key == "x")
                    opts.define_symbol("@@pos", dim);
                else if (key == "v")
                    opts.define_symbol("@@vel", dim);
                opts.define_symbol("@@" + key, dim);
            }
        };

        for (auto &&spgPtr : spgPtrs) {
            // auto &pars = spgPtr->getParticles();
            auto &spg = spgPtr->getSparseGrid();
            typename ZenoSparseGrid::spg_t::grid_storage_type *tvPtr = &spg._grid;

            auto props = tvPtr->getPropertyTags();
            opts.symdims.clear();
            // PropertyTag can be used for structured binding automatically
            for (auto &&[name, nchns] : props)
                def_sym(std::string(name), nchns, spgPtr->isDoubleBufferAttrib(std::string(name)));

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
            bool hasPositionProperty = false;
            /// @note only allow non-double-buffer property insertion
            for (auto const &[name, dim] : prog->newsyms) {
                assert(name[0] == '@');
                auto key = name.substr(1);
                if (key[0] != '@') {
                    if (!checkDuplication(key))
                        newChns.push_back(PropertyTag{key, dim});

                    /// @note currently only check position property among newly inserted symbols
                    if (key == "x" || key == "pos") {
                        hasPositionProperty = true;
                    }
                } else
                    throw std::runtime_error(
                        fmt::format("currently forbids inserting a double buffer property [{}]!", key.substr(1)));
            }
            if (newChns.size() > 0)
                tvPtr->append_channels(cudaPol, newChns);

            // if pos property is accessed, update it
            if (hasPositionProperty) {
                constexpr auto space = execspace_e::cuda;
                if (tvPtr->getPropertySize("x") != 3)
                    throw std::runtime_error("the existing [pos/x] property should be of size 3.");
                cudaPol(range(tvPtr->size()), [voxels = proxy<space>(*tvPtr), posOffset = tvPtr->getPropertyOffset("x"),
                                               spg = proxy<space>(spg)] __device__(int cellno) mutable {
                    voxels.tuple(dim_c<3>, posOffset, cellno) = spg.wCoord(cellno);
                });
            }

            if (_cuModule == nullptr) {
                auto wrangleKernelPtxs = cudri::load_all_ptx_files_at();
                void *state;
                cuLinkCreate(0, nullptr, nullptr, (CUlinkState *)&state);

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
            auto unitBytes = sizeof(RM_CVREF_T(*tvPtr)::value_type);
            constexpr auto tileSize = RM_CVREF_T(*tvPtr)::lane_width;

            /// symbols
            for (int i = 0; i < prog->symbols.size(); i++) {
                auto [name, dimid] = prog->symbols[i];

                int no = -1;
                if (name.substr(0, 2) == "@@") {
                    name = name.substr(2);
                    no = 1;
                } else {
                    name = name.substr(1);
                    no = 0;
                }

                ///@note map reserved keywords
                if (name == "pos")
                    name = "x";
                else if (name == "vel")
                    name = "v";

                ///@note adjust double buffer property name
                if (spgPtr->isDoubleBufferAttrib(name)) {
                    if (no == 1)
                        no = spgPtr->readMeta<int>(name + "_cur") ^ 1;
                    else
                        no = spgPtr->readMeta<int>(name + "_cur");
                } else
                    no = -1;
                if (no >= 0)
                    name += std::to_string(no);

                haccessors[i] = zs::AccessorAoSoA{zs::aosoa_c,
                                                  (void *)tvPtr->data(),
                                                  (unsigned short)unitBytes,
                                                  (unsigned short)tileSize,
                                                  (unsigned short)tvPtr->numChannels(),
                                                  (unsigned short)(tvPtr->getPropertyOffset(name) + dimid),
                                                  (unsigned short)0};

#if 0
        auto t = haccessors[i];
        fmt::print("accessor: numTileBits {} (tileSize {}), {}, {}, "
                   "numUnitBits {} (unitSize {}), {}\n",
                   t.numTileBits, tileSize, t.tileMask, t.chnCnt,
                   t.numUnitSizeBits, unitBytes, t.aux);
        getchar();
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
            std::size_t cnt = tvPtr->size();
            zs::f32 *d_params = dparams.data();
            int nchns = daccessors.size();
            void *addr = daccessors.data();
            void *args[4] = {(void *)&cnt, (void *)&d_params, (void *)&nchns, (void *)&addr};

            cuLaunchKernel((CUfunction)function, (cnt + 127) / 128, 1, 1, 128, 1, 1, 0,
                           (CUstream)currentContext.streamSpare(-1), args, (void **)nullptr);
            // end kernel launch
            cuCtxSynchronize();
        }

        set_output("ZSGrid", get_input("ZSGrid"));
    }

  private:
    void *_cuModule{nullptr};
};

ZENDEFNODE(ZSVolumeWrangler, {
                                 {"ZSGrid", {"string", "zfxCode"}, {"DictObject:NumericObject", "params"}},
                                 {"ZSGrid"},
                                 {},
                                 {"zswrangle"},
                             });

} // namespace zeno
