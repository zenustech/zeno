#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include "zensim/math/bit/Bits.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Relationship.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  namespace mathutil_impl {
    // constexpr scan only available in c++20:
    // https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
    template <typename... Args, std::size_t... Is>
    constexpr auto incl_prefix_sum_impl(std::make_signed_t<std::size_t> I,
                                        std::index_sequence<Is...>, Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is <= I ? std::forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr auto excl_prefix_sum_impl(std::size_t I, std::index_sequence<Is...>,
                                        Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is < I ? std::forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr auto excl_suffix_mul_impl(std::make_signed_t<std::size_t> I,
                                        std::index_sequence<Is...>, Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is > I ? std::forward<Args>(args) : 1) * ...);
    }
  }  // namespace mathutil_impl

  template <typename... Args>
  constexpr auto incl_prefix_sum(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::incl_prefix_sum_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr auto excl_prefix_sum(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::excl_prefix_sum_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr auto excl_suffix_mul(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::excl_suffix_mul_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto incl_prefix_sum(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return incl_prefix_sum(I, Ns...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto excl_prefix_sum(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return excl_prefix_sum(I, Ns...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto excl_suffix_mul(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return excl_suffix_mul(I, Ns...);
  }

  template <typename T, typename Data>
  constexpr auto linear_interop(T &&alpha, Data &&a, Data &&b) noexcept {
    return a + (b - a) * alpha;
  }

  template <typename T, enable_if_t<is_same_v<T, double>> = 0>
  constexpr auto lower_trunc(T v) noexcept {
    return v > 0 ? (i64)v : ((i64)v) - 1;
  }
  template <typename T, enable_if_t<is_same_v<T, float>> = 0>
  constexpr auto lower_trunc(T v) noexcept {
    return v > 0 ? (i32)v : ((i32)v) - 1;
  }

  ///
  /// borrowed from gvdb_library/kernels/cuda_gvdb.cuh
  ///

  constexpr float floatConstruct(uint32_t m) noexcept {
    const uint32_t ieeeMantissa = 0x007FFFFFu;  // binary32 mantissa bitmask
    const uint32_t ieeeOne = 0x3F800000u;       // 1.0 in IEEE binary32
    m &= ieeeMantissa;                          // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                               // Add fractional part to 1.0
    float f = int_as_float(m);                  // Range [1:2]
    return f - 1.0;                             // Range [0:1]
  }

  constexpr uint64_t numBitsOn(uint64_t v) noexcept {
    v = v - ((v >> 1) & 0x5555555555555555LLU);
    v = (v & 0x3333333333333333LLU) + ((v >> 2) & (0x3333333333333333LLU));
    return (((v + (v >> 4)) & 0xF0F0F0F0F0F0F0FLLU) * 0x101010101010101LLU) >> 56;
  }

  constexpr bool isBitOn(uint64_t mask, char b) noexcept {
    return (mask & ((uint64_t)1 << (b & 63))) != 0;
  }

}  // namespace zs
