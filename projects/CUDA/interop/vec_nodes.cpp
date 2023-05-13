#include <tuple>
#include <variant>
#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include "vec.hpp"
#include <fmt/core.h>
namespace zeno
{
struct NumericToSmallVec : INode {
    virtual void apply() override {
        const auto& numVal = get_input<NumericObject>("numeric")->value;
        bool isCol = get_input2<bool>("is_col");
        auto vecType = get_input2<std::string>("vec_type");  
        auto ret = std::make_shared<SmallVecObject>(); 
        std::visit([&ret, vecType](auto const& numObj)
        {
            using num_t = RM_CVREF_T(numObj); 
            if constexpr (zs::is_same_v<num_t, float> || zs::is_same_v<num_t, int>)
            {
                using vec_t = zs::vec<num_t, 1, 1>; 
                ret->value = vec_t{numObj}; 
            }
            else {
                constexpr auto size = std::tuple_size_v<num_t>;  
                using val_t = typename num_t::value_type; 
                if (vecType == "nx1")
                {
                    auto value = zs::vec<val_t, size, 1>(); 
                    for (int d = 0; d < size; d++)
                        value(d, 0) = numObj[d]; 
                    ret->set(value); 
                } else if (vecType == "1xn") {
                    auto value = zs::vec<val_t, 1, size>(); 
                    for (int d = 0; d < size; d++)
                        value(0, d) = numObj[d]; 
                    ret->set(value); 
                } else if (vecType == "n") {
                    auto value = zs::vec<val_t, size>(); 
                    for (int d = 0; d < size; d++)
                        value(d) = numObj[d]; 
                    ret->set(value); 
                }
            }
        }, numVal);  
        set_output("ZSSmallVec", std::move(ret)); 
    }
};

ZENDEFNODE(NumericToSmallVec, {
    {
        "numeric", 
        {"enum n nx1 1xn", "vec_type", "n"}
    },
    {"ZSSmallVec"},
    {}, 
    {"PyZFX"},
});

struct PrintSmallVec : INode
{
    void apply() override {
        const auto& smallVec = get_input<SmallVecObject>("ZSSmallVec")->value;
        std::visit([](auto const &vec) {
            using vec_t = RM_CVREF_T(vec); 
            constexpr auto dim = vec_t::dim; 
            if constexpr (dim == 1)
            {
                fmt::print("PrintSmallVec:\t("); 
                constexpr auto dimI = vec_t::template range_t<0>::value; 
                for (int i = 0; i < dimI; i++)
                    fmt::print("{}, ", vec(i)); 
                fmt::print(")\n"); 
            } else if constexpr (dim == 2)
            {
                constexpr auto dimI = vec_t::template range_t<0>::value; 
                constexpr auto dimJ = vec_t::template range_t<1>::value; 
                fmt::print("PrintSmallVec:[\n");
                for (int i = 0; i < dimI; i++)
                {
                    for (int j = 0; j < dimJ; j++)
                        fmt::print("{}, ", vec(i, j));  
                    fmt::print("\n"); 
                }
                fmt::print("]\n"); 
            }
        }, smallVec); 
    }
}; 
ZENDEFNODE(PrintSmallVec, {
    {"ZSSmallVec"},
    {},
    {}, 
    {"PyZFX"},
});
}