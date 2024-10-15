#include "vec.hpp"
#include <fmt/core.h>
#include <tuple>
#include <variant>
#include <zeno/types/NumericObject.h>
#include <zeno/zeno.h>
namespace zeno {
struct NumericToSmallVec : INode {
    virtual void apply() override {
        const auto &numVal = get_input<NumericObject>("numeric")->value;
        auto vecType = get_input2<std::string>("vec_type");
        auto ret = std::make_shared<SmallVecObject>();
        std::visit(
            [&ret, vecType](auto const &numVal) {
                using num_t = RM_CVREF_T(numVal);
                if constexpr (zs::is_scalar_v<num_t>) {
                    if (vecType == "n") {
                        auto value = zs::vec<num_t, 1>{numVal};
                        ret->set(value);
                    } else if (vecType == "scalar") {
                        ret->set(numVal);
                    } else {
                        auto value = zs::vec<num_t, 1, 1>{numVal};
                        ret->set(value);
                    }
                } else {
                    constexpr auto size = std::tuple_size_v<num_t>;
                    using val_t = typename num_t::value_type;
                    if (vecType == "nx1") {
                        auto value = zs::vec<val_t, size, 1>();
                        for (int d = 0; d < size; d++)
                            value(d, 0) = numVal[d];
                        ret->set(value);
                    } else if (vecType == "1xn") {
                        auto value = zs::vec<val_t, 1, size>();
                        for (int d = 0; d < size; d++)
                            value(0, d) = numVal[d];
                        ret->set(value);
                    } else if (vecType == "n") {
                        auto value = zs::vec<val_t, size>();
                        for (int d = 0; d < size; d++)
                            value(d) = numVal[d];
                        ret->set(value);
                    } else {
                        auto errorMsg = fmt::format("Cannot convert to zs small vec of type {}", vecType);
                        throw std::runtime_error(errorMsg);
                    }
                }
            },
            numVal);
        set_output("ZSSmallVec", std::move(ret));
    }
};

ZENDEFNODE(NumericToSmallVec, {
                                  {"numeric", {"enum n nx1 1xn scalar", "vec_type", "n"}},
                                  {"ZSSmallVec"},
                                  {},
                                  {"PyZFX"},
                              });

struct SmallVecToNumeric : INode {
    virtual void apply() override {
        const auto &smallVec = get_input<SmallVecObject>("ZSSmallVec")->value;
        auto ret = std::make_shared<NumericObject>();
        std::visit(
            [&ret](auto const &vec) {
                using vec_t = RM_CVREF_T(vec);
                using VT = zs::conditional_t<zs::is_same_v<vec_t, double>, float, vec_t>;
                if constexpr (zs::is_scalar_v<vec_t>) {
                    ret->set((VT)vec);
                } else {
                    constexpr auto dim = vec_t::dim;
                    using VT = zs::conditional_t<zs::is_same_v<typename vec_t::value_type, double>, 
                        float, typename vec_t::value_type>;
                    if constexpr (dim == 1) {
                        constexpr auto dimI = vec_t::template range_t<0>::value;
                        if constexpr (dimI == 1) {
                            ret->set((VT)vec(0));
                        } else {
                            zeno::vec<dimI, VT> tmp;
                            for (int d = 0; d < dimI; ++d)
                                tmp[d] = vec(d);
                            ret->set(tmp);
                        }
                    } else if constexpr (dim == 2) {
                        constexpr auto dimI = vec_t::template range_t<0>::value;
                        constexpr auto dimJ = vec_t::template range_t<1>::value;
                        if constexpr (dimI == 1) {
                            if constexpr (dimJ <= 4) {
                                if constexpr (dimJ == 1) {
                                    ret->set((VT)vec(0, 0));
                                } else {
                                    zeno::vec<dimJ, VT> tmp;
                                    for (int d = 0; d < dimJ; ++d)
                                        tmp[d] = vec(0, d);
                                    ret->set(tmp);
                                }
                            } else {
                                static_assert(zs::always_false<vec_t>, "...");
                            }
                        } else if constexpr (dimJ == 1) {
                            if constexpr (dimI <= 4) {
                                if constexpr (dimI == 1) {
                                    ret->set((VT)vec(0, 0));
                                } else {
                                    zeno::vec<dimI, VT> tmp;
                                    for (int d = 0; d < dimI; ++d)
                                        tmp[d] = vec(d, 0);
                                    ret->set(tmp);
                                }
                            } else {
                                static_assert(zs::always_false<vec_t>, "...");
                            }
                        } else {
                            throw std::runtime_error(fmt::format(
                                "cannot convert a small vec of shape ({}, {}) to zeno NumericValue", dimI, dimJ));
                        }
                    }                    
                }
            },
            smallVec);
        set_output("numeric", std::move(ret));
    }
};

ZENDEFNODE(SmallVecToNumeric, {
                                  {"ZSSmallVec"},
                                  {"numeric"},
                                  {},
                                  {"PyZFX"},
                              });

struct PrintSmallVec : INode {
    void apply() override {
        const auto &smallVec = get_input<SmallVecObject>("ZSSmallVec")->value;
        std::visit(
            [](auto const &vec) {
                using vec_t = RM_CVREF_T(vec);
                if constexpr (zs::is_scalar_v<vec_t>) {
                    fmt::print("PrintSmallVec:\t{}\n", vec); 
                } else {
                    constexpr auto dim = vec_t::dim;
                    if constexpr (dim == 1) {
                        fmt::print("PrintSmallVec:\t(");
                        constexpr auto dimI = vec_t::template range_t<0>::value;
                        for (int i = 0; i < dimI - 1; i++)
                            fmt::print("{}, ", vec(i));
                        fmt::print("{})\n", vec(dimI - 1));
                    } else if constexpr (dim == 2) {
                        constexpr auto dimI = vec_t::template range_t<0>::value;
                        constexpr auto dimJ = vec_t::template range_t<1>::value;
                        fmt::print("PrintSmallVec:[\n");
                        for (int i = 0; i < dimI; i++) {
                            fmt::print("\t"); 
                            for (int j = 0; j < dimJ; j++)
                                fmt::print(((i == dimI - 1) && (j == dimJ - 1)) ? "{}" : "{}, ", vec(i, j));
                            fmt::print("\n");
                        }
                        fmt::print("]\n");
                    }                    
                }
            },
            smallVec);
    }
};
ZENDEFNODE(PrintSmallVec, {
                              {"ZSSmallVec"},
                              {},
                              {},
                              {"PyZFX"},
                          });
} // namespace zeno