#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include "macros.h"

namespace zeno::rxmesh {

struct LocalIndexT {
    __device__ __host__ LocalIndexT() : id(INVALID16) {}

    __device__ __host__ LocalIndexT(uint16_t id) : id(id) {}
    uint16_t id;
};
namespace detail {
/**
 * @brief hash function that takes a pair of vertices and returns a unique
 * values. Used for storing vertex-edge relation in std map
 */
struct edge_key_hash {
    // www.techiedelight.com/use-std-pair-key-std-unordered_map-cpp/
    template <class T>
    inline std::size_t operator()(const std::pair<T, T>& e_key) const {
        return std::hash<T>()(e_key.first * 8191 + e_key.second * 11003);
    }
};

inline std::pair<uint32_t, uint32_t> edge_key(const uint32_t v0,
                                              const uint32_t v1) {
    uint32_t i = std::max(v0, v1);
    uint32_t j = std::min(v0, v1);
    return std::make_pair(i, j);
}

/**
 * @brief extracting the input parameter type and return type of a lambda
 * function. Taken from https://stackoverflow.com/a/7943765/1608232.
 * For generic types, directly use the result of the signature of its operator()
 */
template <typename T>
struct FunctionTraits : public FunctionTraits<decltype(&T::operator())>{};

/**
 * @brief specialization for pointers to member function
 */
template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const> {
    /**
     * @brief arity is the number of arguments.
     */
    enum {
        arity = sizeof...(Args)
    };

    typedef ReturnType result_type;

    /**
     * @brief the i-th argument is equivalent to the i-th tuple element of a
     * tuple composed of those arguments.
     */
    template <size_t i>
    struct arg {
        using type_rc =
            typename std::tuple_element<i, std::tuple<Args...>>::type;
        using type_c = std::conditional_t<std::is_reference_v<type_rc>,
                                          std::remove_reference_t<type_rc>,
                                          type_rc>;
        using type   = std::conditional_t<std::is_const_v<type_c>,
                                        std::remove_const_t<type_c>,
                                        type_c>;
    };
};

}  // namespace detail
}  // namespace rxmesh