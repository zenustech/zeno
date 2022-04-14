#pragma once

#include <execution>

namespace zeno {

#ifdef ZENO_PARALLEL_STL
using sequenced_policy = std::execution::sequenced_policy;
using parallel_policy = std::execution::parallel_policy;
using parallel_unsequenced_policy = std::execution::parallel_unsequenced_policy;
static inline constexpr auto seq = std::execution::seq;
static inline constexpr auto par = std::execution::par;
static inline constexpr auto par_unseq = std::execution::par_unseq;
#else
using sequenced_policy = std::execution::sequenced_policy;
using parallel_policy = std::execution::sequenced_policy;
using parallel_unsequenced_policy = std::execution::sequenced_policy;
static inline constexpr auto seq = std::execution::seq;
static inline constexpr auto par = std::execution::seq;
static inline constexpr auto par_unseq = std::execution::seq;
#endif

}
