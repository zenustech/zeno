#include "TileVector.hpp"
#include "zensim/ZpcFunctional.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include <fmt/core.h>
#include <tuple>
#include <variant>
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
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

} // namespace zeno