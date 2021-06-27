#pragma once
/// be cautious to include this header
/// to enable cuda compiler, include cuda header before this one

/// use these functions within other templated function (1) or in a source file (2)
/// (1)
/// REMEMBER! Make Sure Their Specializations Done In the Correct Compiler Context!
/// which is given a certain execution policy tag, necessary headers are to be included
/// (2)
/// inside a certain source file

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/bit/Bits.h"
#if defined(_WIN32)
#include <intrin.h>
#include <stdlib.h>
#endif

namespace zs {

#if !ZS_ENABLE_CUDA || !defined(__CUDACC__)
#  define __device__ 
#endif

  // __threadfence
  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  inline __device__ void thread_fence(ExecTag) {
#if defined(__CUDACC__)
    __threadfence();
#else
    throw std::runtime_error(
        fmt::format("thread_fence(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
#endif
  }

  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, omp_exec_tag>> = 0>
  inline void thread_fence(ExecTag) noexcept {
    /// a thread is guaranteed to see a consistent view of memory with respect to the variables in “
    /// list ”
#pragma omp flush
  }

  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, host_exec_tag>> = 0>
  inline void thread_fence(ExecTag) noexcept {}

  // __activemask
  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  inline __device__ unsigned active_mask(ExecTag) {
#if defined(__CUDACC__)
    return __activemask();
#else
    throw std::runtime_error(
        fmt::format("active_mask(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
#endif
  }

#if 0
  template <typename ExecTag, enable_if_t<!is_same_v<ExecTag, cuda_exec_tag>> = 0>
  unsigned active_mask(ExecTag) noexcept {
    return ~0u;
  }
#endif

  // __ballot_sync
  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  inline __device__ unsigned ballot_sync(ExecTag, unsigned mask, int predicate) {
#if defined(__CUDACC__)
    return __ballot_sync(mask, predicate);
#else
    throw std::runtime_error(
        fmt::format("ballot_sync(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
#endif
  }

#if 0
  template <typename ExecTag, enable_if_t<!is_same_v<ExecTag, cuda_exec_tag>> = 0>
  unsigned ballot_sync(ExecTag, unsigned mask, int predicate) noexcept {
    return ~0u;
  }
#endif

  // ref: https://graphics.stanford.edu/~seander/bithacks.html

  /// count leading zeros
  template <typename ExecTag, typename T, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  inline __device__ int count_lz(ExecTag, T x) {
#if defined(__CUDACC__)
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(int) == nbytes)
      return __clz((int)x);
    else if constexpr (sizeof(long long int) == nbytes)
      return __clzll((long long int)x);
#endif
    throw std::runtime_error(fmt::format("count_lz(tag {}, {} bytes) not viable\n",
                                         get_execution_space_tag(ExecTag{}), sizeof(T)));
  }
  template <typename ExecTag, typename T, enable_if_t<is_same_v<ExecTag, host_exec_tag>> = 0>
  constexpr int count_lz(ExecTag, T x) {
    return (int)count_leading_zeros(x);
  }
  template <typename ExecTag, typename T, enable_if_t<is_same_v<ExecTag, omp_exec_tag>> = 0>
  constexpr int count_lz(ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if (x == (T)0) return nbytes * 8;
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    if constexpr (sizeof(unsigned short) == nbytes)
      return __lzcnt16((unsigned short)x);
    else if constexpr (sizeof(unsigned int) == nbytes)
      return __lzcnt((unsigned int)x);
    else if constexpr (sizeof(unsigned __int64) == nbytes)
      return __lzcnt64((unsigned __int64)x);
#elif defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(unsigned int) == nbytes)
      return __builtin_clz((unsigned int)x);
    else if constexpr (sizeof(unsigned long) == nbytes)
      return __builtin_clzl((unsigned long)x);
    else if constexpr (sizeof(unsigned long long) == nbytes)
      return __builtin_clzll((unsigned long long)x);
#endif
    throw std::runtime_error(fmt::format("count_lz(tag {}, {} bytes) not viable\n",
                                         get_execution_space_tag(ExecTag{}), sizeof(T)));
  }

  /// reverse bits
  template <typename ExecTag, typename T, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  inline __device__ T reverse_bits(ExecTag, T x) {
#if defined(__CUDACC__)
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(unsigned int) == nbytes)
      return __brev((unsigned int)x);
    else if constexpr (sizeof(unsigned long long int) == nbytes)
      return __brevll((unsigned long long int)x);
#endif
    throw std::runtime_error(fmt::format("reverse_bits(tag {}, {} bytes) not viable\n",
                                         get_execution_space_tag(ExecTag{}), sizeof(T)));
  }
  template <typename ExecTag, typename T, enable_if_t<is_same_v<ExecTag, host_exec_tag>> = 0>
  constexpr T reverse_bits(ExecTag, T x) {
    return binary_reverse(x);
  }
  template <typename ExecTag, typename T, enable_if_t<is_same_v<ExecTag, omp_exec_tag>> = 0>
  constexpr T reverse_bits(ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if (x == (T)0) return 0;
    using Val = std::make_unsigned_t<T>;
    Val tmp{}, ret{0};
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    if constexpr (sizeof(unsigned short) == nbytes)
      tmp = (Val)_byteswap_ushort((unsigned short)x);
    else if constexpr (sizeof(unsigned long) == nbytes)
      tmp = (Val)_byteswap_ulong((unsigned long)x);
    else if constexpr (sizeof(unsigned __int64) == nbytes)
      tmp = (Val)_byteswap_uint64((unsigned __int64)x);
#elif defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(unsigned short) == nbytes)
      tmp = (Val)__builtin_bswap16((unsigned short)x);
    else if constexpr (sizeof(unsigned int) == nbytes)
      tmp = (Val)__builtin_bswap32((unsigned int)x);
    else if constexpr (sizeof(unsigned long long) == nbytes)
      tmp = (Val)__builtin_bswap64((unsigned long long)x);
#endif
    else
      throw std::runtime_error(fmt::format("reverse_bits(tag {}, {} bytes) not viable\n",
                                           get_execution_space_tag(ExecTag{}), sizeof(T)));
    // reverse within each byte
    for (int bitoffset = 0; tmp; bitoffset += 8) {
      unsigned char b = tmp & 0xff;
      b = ((u64)b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
      ret |= ((Val)b << bitoffset);
      tmp >>= 8;
    }
    return (T)ret;
  }

#if !ZS_ENABLE_CUDA || !defined(__CUDACC__)
#  undef __device__ 
#endif

}  // namespace zs