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

// NOTE: assume this tv is on the device for now
template <class TV>
typename TV::value_type extractScalarFromTileVector(const TV &tv, zs::SmallString tagName, int dim, int index) {
    auto pol = zs::cuda_exec();
    using tv_t = RM_CVREF_T(tv);
    using val_t = typename tv_t::value_type;
    zs::Vector<val_t> res{1, zs::memsrc_e::device, 0};
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(range(1), [tv = proxy<space>({}, tv), res = proxy<space>(res), tagName = SmallString{tagName}, index = index,
                   dim = dim] __device__(int i) mutable { res[0] = tv(tagName, dim, index); });
    return res.getVal();
}

namespace zeno {
struct MakeZsTileVector : INode {
    void apply() override {
        auto input_size = get_input2<int>("size");
        auto input_memsrc = get_input2<std::string>("memsrc");
        auto input_prop_tags = get_input<DictObject>("prop_tags")->getLiterial<int>();
        auto intput_devid = get_input2<int>("dev_id");
        auto intput_elem_type = get_input2<std::string>("elem_type");

        zs::memsrc_e memsrc;
        if (input_memsrc == "host")
            memsrc = zs::memsrc_e::host;
        else if (input_memsrc == "device")
            memsrc = zs::memsrc_e::device;
        else
            memsrc = zs::memsrc_e::um;

        std::vector<zs::PropertyTag> tags;
        for (auto const &[k, v] : input_prop_tags)
            tags.push_back({k, v});

#define MAKE_TILEVECTOR_OBJ_T(T)                                                               \
    if (intput_elem_type == #T) {                                                              \
        auto allocator = zs::get_memory_source(memsrc, static_cast<zs::ProcID>(intput_devid)); \
        tvObj->set(zs::TileVector<T, 32, zs::ZSPmrAllocator<false>>{allocator, tags, 0});      \
    }

        auto tvObj = std::make_shared<ZsTileVectorObject>();
        MAKE_TILEVECTOR_OBJ_T(int)
        MAKE_TILEVECTOR_OBJ_T(float)
        MAKE_TILEVECTOR_OBJ_T(double)
        std::visit([input_size](auto &tv) { tv.resize(input_size); }, tvObj->value);

        set_output("ZsTileVector", std::move(tvObj));
    }
};

ZENDEFNODE(MakeZsTileVector, {
                                 {{"int", "size", "0"},
                                  {"DictObject", "prop_tags"},
                                  {"enum host device um", "memsrc", "device"},
                                  {"int", "dev_id", "0"},
                                  {"enum float int double", "elem_type", "float"}},
                                 {"ZsTileVector"},
                                 {},
                                 {"PyZFX"},
                             });

struct ExtractScalarFromZsTileVector : INode {
    void apply() override {
        auto tvObj = get_input<ZsTileVectorObject>("ZsTileVector");
        auto index = get_input2<int>("index");
        auto tagName = get_input2<std::string>("prop_tag");
        auto dim = get_input2<int>("dim");

        auto &tv = tvObj->value;
        auto result = std::make_shared<NumericObject>();
        std::visit(
            [&](auto &tv) {
                if (tv.memspace() == zs::memsrc_e::device) {
                    auto val = extractScalarFromTileVector(tv, tagName, dim, index);
                    using val_t = RM_CVREF_T(val);
                    if constexpr (zs::is_same_v<val_t, int>)
                        result->set(val);
                    else
                        result->set(static_cast<float>(val));
                } else {
                    using namespace zs;
                    constexpr auto space = zs::execspace_e::host;
                    auto view = proxy<space>({}, tv);
                    auto val = view(tagName, dim, index);
                    using val_t = RM_CVREF_T(val);
                    if constexpr (zs::is_same_v<val_t, int>)
                        result->set(val);
                    else
                        result->set(static_cast<float>(val));
                }
            },
            tv);
        set_output2("result", std::move(result));
    }
};

ZENDEFNODE(ExtractScalarFromZsTileVector,
           {
               {"ZsTileVector", {"int", "index", "0"}, {"string", "prop_tag", ""}, {"int", "dim", "0"}},
               {"result"},
               {},
               {"PyZFX"},
           });

struct CopyZsTileVectorTo : INode {
    template <typename SrcRange, typename DstRange, int dim, bool use_bit_cast>
    static void rearrange_device_data(SrcRange &&srcRange, DstRange &&dstRange, zs::wrapv<dim>,
                                      zs::wrapv<use_bit_cast>) {
        using namespace zs;
        using SrcT = RM_CVREF_T(*zs::begin(srcRange));
        using DstT = RM_CVREF_T(*zs::begin(dstRange));
        cuda_exec()(zip(srcRange, dstRange), [] __device__(SrcT src, DstT & dst) {
            if constexpr (use_bit_cast) {
                if constexpr (is_arithmetic_v<DstT>)
                    dst = reinterpret_bits<DstT>(src);
                else {
                    static_assert(is_vec_v<SrcT>, "expect zs small vec here!");
                    dst = src.reinterpret_bits(wrapt<typename DstT::value_type>{});
                }
            } else
                dst = src;
        });
    }
    void apply() override {
        auto tvObj = get_input<ZsTileVectorObject>("ZsTileVector");
        auto prim = get_input<PrimitiveObject>("prim");
        auto attr = get_input2<std::string>("attr");
        auto &tv = tvObj->value;

        std::visit(
            [&prim, &attr](auto &tv) {
                using tv_t = RM_CVREF_T(tv);
                using val_t = typename tv_t::value_type;
                using namespace zs;
                if constexpr (zs::is_arithmetic_v<val_t>) {
                    if (prim->size() != tv.size()) {
                        fmt::print("BEWARE! copy sizes mismatch! resize to match.\n");
                        prim->resize(tv.size());
                    }

                    match([&tv, &attr](auto &primAttrib) {
                        using T = typename RM_CVREF_T(primAttrib)::value_type;
                        if constexpr (zs::is_arithmetic_v<T>) {
                            using AllocatorT = RM_CVREF_T(tv.get_allocator());
                            zs::Vector<T, AllocatorT> stage{tv.get_allocator(), tv.size()};
                            if (tv.memoryLocation().onHost()) {
                                /// T and val_t may diverge
                                omp_exec()(zip(range(tv, attr, value_seq<1>{}, wrapt<val_t>{}), stage),
                                           [](val_t src, T &dst) { dst = src; });
                                std::memcpy(primAttrib.data(), stage.data(), sizeof(T) * tv.size());
                            } else {
                                rearrange_device_data(range(tv, attr, value_seq<1>{}, wrapt<val_t>{}), range(stage),
                                                      wrapv<1>{}, false_c);
                                zs::copy(mem_device, (void *)primAttrib.data(), (void *)stage.data(),
                                         sizeof(T) * tv.size());
                            }
                        } else {
                            using TT = typename T::value_type;
                            constexpr int dim = std::tuple_size_v<T>;
                            using ZsT = zs::vec<TT, dim>;
                            static_assert(sizeof(T) == sizeof(ZsT) && alignof(T) == alignof(ZsT),
                                          "corresponding zs element type dudection failed.");
                            using AllocatorT = RM_CVREF_T(tv.get_allocator());
                            zs::Vector<ZsT, AllocatorT> stage{tv.get_allocator(), tv.size()};
                            if (tv.memoryLocation().onHost()) {
                                /// T and val_t may diverge
                                omp_exec()(zip(range(tv, attr, value_seq<dim>{}, wrapt<val_t>{}), stage),
                                           [dim = dim](auto src, ZsT &dst) {
                                               for (int d = 0; d != dim; ++d)
                                                   dst[d] = src[d];
                                           });
                                std::memcpy(primAttrib.data(), stage.data(), sizeof(T) * tv.size());
                            } else {
                                rearrange_device_data(range(tv, attr, value_seq<dim>{}, wrapt<val_t>{}), range(stage),
                                                      wrapv<dim>{}, false_c);
                                zs::copy(mem_device, (void *)primAttrib.data(), (void *)stage.data(),
                                         sizeof(ZsT) * tv.size());
                            }
                        }
                    })(prim->attr(attr));

                } else
                    throw std::runtime_error("unable to copy tilevector of non-arithmetic value_type yet");
            },
            tv);

        set_output2("prim", prim);
    }
};

ZENDEFNODE(CopyZsTileVectorTo, {
                                   {"ZsTileVector",
                                    {"PrimitiveObject", "prim"},
                                    {"string", "attr", "clr"},
                                    {"enum convert enforce_bit_cast", "option", "convert"}},
                                   {"prim"},
                                   {},
                                   {"PyZFX"},
                               });

} // namespace zeno