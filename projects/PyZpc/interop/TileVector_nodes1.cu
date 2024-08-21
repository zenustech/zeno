#include "TileVector.hpp"
#include "zensim/ZpcFunctional.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <fmt/core.h>
#include <tuple>
#include <variant>
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>

namespace zeno {
    struct CopyZsTileVectorFrom : INode {
        template <typename SrcRange, typename DstRange, bool use_bit_cast>
        static void rearrange_device_data(SrcRange&& srcRange, DstRange&& dstRange, zs::wrapv<use_bit_cast>) {
            using namespace zs;
            cuda_exec()(zip(srcRange, dstRange), [] __device__(const auto & src, auto && dst) {
                using SrcT = RM_CVREF_T(src);
                using DstT = RM_CVREF_T(dst);
                if constexpr (use_bit_cast) {
                    if constexpr (is_arithmetic_v<SrcT>) {
                        dst = reinterpret_bits<DstT>(src);
                    }
                    else {
                        static_assert(is_vec_v<SrcT>, "expect zs small vec here!");
                        dst = src.reinterpret_bits(wrapt<typename DstT::value_type>{});
                    }
                }
                else
                    dst = src;
            });
        }
        void apply() override {
            auto tvObj = get_input<ZsTileVectorObject>("ZsTileVector");
            auto prim = get_input<PrimitiveObject>("prim");
            auto attr = get_input2<std::string>("attr");
            auto& tv = tvObj->value;

            std::visit(
                [&prim, &attr](auto& tv) {
                    using tv_t = RM_CVREF_T(tv);
                    using val_t = typename tv_t::value_type;
                    using namespace zs;
                    if constexpr (zs::is_arithmetic_v<val_t>) {
                        if (prim->size() != tv.size()) {
                            fmt::print("BEWARE! copy sizes mismatch! resize to match.\n");
                            tv.resize(prim->size());
                        }

                        auto process = [&tv, &attr](auto& primAttrib) {
                            using T = typename RM_CVREF_T(primAttrib)::value_type;
                            if constexpr (zs::is_arithmetic_v<T>) {
                                using AllocatorT = RM_CVREF_T(tv.get_allocator());
                                zs::Vector<T, AllocatorT> stage{ tv.size() };
                                std::memcpy(stage.data(), primAttrib.data(), sizeof(T) * tv.size());
                                if (tv.memoryLocation().onHost()) {
                                    /// T and val_t may diverge
                                    omp_exec()(zip(stage, range(tv, attr, value_seq<1>{}, wrapt<val_t>{})),
                                        [](T src, val_t& dst) { dst = src; });
                                }
                                else {
                                    stage = stage.clone(tv.memoryLocation());
                                    rearrange_device_data(range(stage), range(tv, attr, value_seq<1>{}, wrapt<val_t>{}),
                                        false_c);
                                }
                            }
                            else {
                                puts("0");
                                using TT = typename T::value_type;
                                constexpr int dim = std::tuple_size_v<T>;
                                using ZsT = zs::vec<TT, dim>;
                                static_assert(sizeof(T) == sizeof(ZsT) && alignof(T) == alignof(ZsT),
                                    "corresponding zs element type dudection failed.");
                                using AllocatorT = RM_CVREF_T(tv.get_allocator());
                                zs::Vector<ZsT, AllocatorT> stage{ tv.size() };
                                std::memcpy(stage.data(), primAttrib.data(), sizeof(T) * tv.size());
                                if (tv.memoryLocation().onHost()) {
                                    /// T and val_t may diverge
                                    omp_exec()(zip(stage, range(tv, attr, value_seq<dim>{}, wrapt<val_t>{})),
                                        [dim = dim](const ZsT& src, auto& dst) {
                                            for (int d = 0; d != dim; ++d)
                                                dst[d] = src[d];
                                        });
                                }
                                else {
                                    stage = stage.clone(tv.memoryLocation());
                                    rearrange_device_data(stage, range(tv, attr, value_seq<dim>{}, wrapt<val_t>{}),
                                        false_c);
                                }
                            }
                            };
                        if (attr == "pos")
                            // if constexpr (zs::is_same_v<std::vector<zeno::vec3f>, RM_CVREF_T(prim->attr(attr))>)
                            process(prim->attr<zeno::vec3f>("pos"));
                        else
                            match(process)(prim->attr(attr));

                    }
                    else
                        throw std::runtime_error("unable to copy tilevector of non-arithmetic value_type yet");
                },
                tv);

            set_output2("ZsTileVector", tvObj);
        }
    };

    ZENDEFNODE(CopyZsTileVectorFrom, {
                                         {"ZsTileVector",
                                          {gParamType_Primitive, "prim"},
                                          {gParamType_String, "attr", "clr"},
                                          {"enum convert enforce_bit_cast", "option", "convert"}},
                                         {"ZsTileVector"},
                                         {},
                                         {"PyZFX"},
        });

} // namespace zeno