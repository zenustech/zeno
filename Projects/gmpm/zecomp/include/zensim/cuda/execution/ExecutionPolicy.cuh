#pragma once

#include <cooperative_groups.h>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "zensim/cuda/Cuda.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/types/Tuple.h"
// #include <device_types.h>
#include <iterator>
#include <nvfunctional>
#include <type_traits>

#include "zensim/resource/Resource.h"
#include "zensim/types/Function.h"

/// extracted from compiler error message...
template <class Tag, class... CapturedVarTypePack> struct __nv_dl_wrapper_t;
template <class U, U func, unsigned int> struct __nv_dl_tag;

namespace zs {

  // =========================  signature  ==============================
  // loopbody signature: (blockid, warpid, threadid, scratchpadMemory)
  template <typename Tn, typename F> __global__ void thread_launch(Tn n, F f) {
    extern __shared__ char shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = function_traits<F>;
      if constexpr (func_traits::arity == 1)
        f(id);
      else if constexpr (func_traits::arity == 2
                         && std::is_pointer_v<
                             std::tuple_element_t<0, typename func_traits::arguments_t>>)
        f(shmem, id);
    }
  }
  template <typename F> __global__ void block_thread_launch(F f) {
    extern __shared__ char shmem[];
    using func_traits = function_traits<F>;
    if constexpr (func_traits::arity == 2
                  && !std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(blockIdx.x, threadIdx.x);
    else if constexpr (func_traits::arity == 3
                       && std::is_pointer_v<
                           std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(shmem, blockIdx.x, threadIdx.x);
  }
  template <bool withIndex, typename Tn, typename F, typename Tuple, std::size_t... Is>
  __forceinline__ __device__ void range_foreach(std::bool_constant<withIndex>, Tn i, F&& f, Tuple args, std::index_sequence<Is...>) {
    if constexpr (withIndex)
      f(i, deref_if_raiter(args.template get<Is>(), i)...);
    else
      f(deref_if_raiter(args.template get<Is>(), i)...);
  }
  template <bool withIndex, typename Tn, typename F, typename Tuple, std::size_t... Is>
  __forceinline__ __device__ void range_foreach(std::bool_constant<withIndex>, char *shmem, Tn i, F&& f, Tuple args, std::index_sequence<Is...>) {
    if constexpr (withIndex)
      f(shmem, i, deref_if_raiter(args.template get<Is>(), i)...);
    else
      f(shmem, deref_if_raiter(args.template get<Is>(), i)...);
  }
  template <typename Tn, typename F, typename... IterOrVals> __global__ void range_launch(Tn n, F f, IterOrVals... iterOrVals) {
    extern __shared__ char shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = function_traits<F>;
      constexpr auto numArgs = sizeof...(IterOrVals);

      if constexpr (func_traits::arity == numArgs)
        range_foreach(std::false_type{}, id, f, zs::forward_as_tuple(iterOrVals...), std::index_sequence_for<IterOrVals...>{});
      else if constexpr (func_traits::arity == numArgs + 1) {
        if constexpr (std::is_integral_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
          range_foreach(std::true_type{}, id, f, zs::forward_as_tuple(iterOrVals...), std::index_sequence_for<IterOrVals...>{});
        else if constexpr (std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
          range_foreach(std::false_type{}, shmem, id, f, zs::forward_as_tuple(iterOrVals...), std::index_sequence_for<IterOrVals...>{});
      }
      else if constexpr (func_traits::arity == numArgs + 2 && std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
        range_foreach(std::true_type{}, shmem, id, f, zs::forward_as_tuple(iterOrVals...), std::index_sequence_for<IterOrVals...>{});
    }
  }
  namespace cg = cooperative_groups;
  template <typename Tn, typename F> __global__ void block_tile_lane_launch(Tn tileSize, F f) {
    extern __shared__ char shmem[];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_group tile = cg::tiled_partition(block, tileSize);
    using func_traits = function_traits<F>;
    if constexpr (func_traits::arity == 3
                  && !std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
    else if constexpr (func_traits::arity == 4
                       && std::is_pointer_v<
                           std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(shmem, blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
  }

  namespace detail {
    template <typename, typename> struct function_traits_impl;

    template <auto F, unsigned int I, typename R, typename... Args>
    struct function_traits_impl<__nv_dl_tag<R (*)(Args...), F, I>> {
      static constexpr std::size_t arity = sizeof...(Args);
      using return_t = R;
      using arguments_t = std::tuple<Args...>;
    };
    template <class Tag, class... CapturedVarTypePack>
    struct function_traits_impl<__nv_dl_wrapper_t<Tag, CapturedVarTypePack...>>
        : function_traits_impl<Tag> {};
  }  // namespace detail

  struct CudaExecutionPolicy : ExecutionPolicyInterface<CudaExecutionPolicy> {
    CudaExecutionPolicy &listen(ProcID incProc, StreamID incStream) {
      this->_wait = true;
      incomingProc = incProc;
      incomingStreamid = incStream;
      return *this;
    }
    CudaExecutionPolicy &stream(StreamID streamid_) {
      streamid = streamid_;
      return *this;
    }
    CudaExecutionPolicy &device(ProcID pid) {
      procid = pid;
      return *this;
    }
    CudaExecutionPolicy &shmem(std::size_t bytes) {
      shmemBytes = bytes;
      return *this;
    }
#if 0
    template <typename FTraits> static constexpr unsigned computeArity() noexcept {
      unsigned res{0};
      if constexpr (FTraits::arity != 0)
        res = FTraits::arity
              - (std::is_pointer_v<std::tuple_element_t<0, typename FTraits::arguments_t>> ? 1 : 0);
      return res;
    }
#endif
    template <typename Tn, typename F> void operator()(std::initializer_list<Tn> dims, F &&f) const {
      const std::vector<Tn> range{dims};
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));
      // need to work on __device__ func as well
      // if constexpr (arity == 1)
      if (range.size() == 1)
        context.launchSpare(streamid, {(range[0] + 127) / 128, 128, shmemBytes}, thread_launch,
                            range[0], f);
      // else if constexpr (arity == 2)
      else if (range.size() == 2)
        context.launchSpare(streamid, {range[0], range[1], shmemBytes}, block_thread_launch, f);
      // else if constexpr (arity == 3)
      else if (range.size() == 3)
        context.launchSpare(streamid, {range[0], range[1] * range[2], shmemBytes},
                            block_tile_lane_launch, range[2], f);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <typename Range, typename F> void operator()(Range &&range, F &&f) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));

      // need to work on __device__ func as well
      auto iter = std::begin(range);
      using IterT = remove_cvref_t<decltype(iter)>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      const DiffT dist = std::end(range) - iter;

      context.launchSpare(streamid, {(dist + 127) / 128, 128, shmemBytes}, range_launch, dist, f, iter);

      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }

    /// for_each
    template <class ForwardIt, class UnaryFunction>
    void for_each_impl(std::random_access_iterator_tag, ForwardIt &&first, ForwardIt &&last,
                       UnaryFunction &&f) const {
      using IterT = remove_cvref_t<ForwardIt>;
      const auto dist = last - first;
      (*this)({dist}, [first, f](typename std::iterator_traits<IterT>::difference_type tid) {
        f(*(first + tid));
      });
    }
    template <class ForwardIt, class UnaryFunction>
    void for_each(ForwardIt &&first, ForwardIt &&last, UnaryFunction &&f) const {
      for_each_impl(typename std::iterator_traits<remove_cvref_t<ForwardIt>>::iterator_category{},
                    FWD(first), FWD(last), FWD(f));
    }
    /// inclusive scan
    template <class InputIt, class OutputIt, class BinaryOperation>
    void inclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                             OutputIt &&d_first, BinaryOperation &&binary_op) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, first.operator->(),
                                     d_first.operator->(), binary_op, dist,
                                     (cudaStream_t)context.streamSpare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceScan::InclusiveScan(d_tmp, temp_storage_bytes, first, d_first, binary_op, dist,
                                     (cudaStream_t)context.streamSpare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <class InputIt, class OutputIt,
              class BinaryOperation = std::plus<remove_cvref_t<decltype(*std::declval<InputIt>())>>>
    void inclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                        BinaryOperation &&binary_op = {}) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      inclusive_scan_impl(
          typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{}, FWD(first),
          FWD(last), FWD(d_first), FWD(binary_op));
    }
    /// exclusive scan
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    void exclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                             OutputIt &&d_first, T init, BinaryOperation &&binary_op) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, first, d_first, binary_op, init,
                                     dist, (cudaStream_t)context.streamSpare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes, first, d_first, binary_op, init,
                                     dist, (cudaStream_t)context.streamSpare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
              class BinaryOperation = std::plus<T>>
    void exclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                        T init = monoid_op<BinaryOperation>::e,
                        BinaryOperation &&binary_op = {}) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      exclusive_scan_impl(
          typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{}, FWD(first),
          FWD(last), FWD(d_first), init, FWD(binary_op));
    }
    /// reduce
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    void reduce_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                     OutputIt &&d_first, T init, BinaryOperation &&binary_op) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, first, d_first, dist, binary_op, init,
                                (cudaStream_t)context.streamSpare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceReduce::Reduce(d_tmp, temp_storage_bytes, first, d_first, dist, binary_op, init,
                                (cudaStream_t)context.streamSpare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
              class BinaryOp = std::plus<T>>
    void reduce(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                T init = monoid_op<BinaryOp>::e, BinaryOp &&binary_op = {}) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      reduce_impl(typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{},
                  FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op));
    }
    /// histogram sort
    /// radix sort pair
    template <class KeyIter, class ValueIter,
              typename Tn = typename std::iterator_traits<remove_cvref_t<KeyIter>>::difference_type>
    std::enable_if_t<std::is_convertible_v<
        typename std::iterator_traits<remove_cvref_t<KeyIter>>::iterator_category,
        std::random_access_iterator_tag>>
    radix_sort_pair(KeyIter &&keysIn, ValueIter &&valsIn, KeyIter &&keysOut, ValueIter &&valsOut,
                    Tn count = 0, int sbit = 0,
                    int ebit
                    = sizeof(typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type)
                      * 8) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));
      if (count) {
        std::size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, keysIn.operator->(),
                                        keysOut.operator->(), valsIn.operator->(),
                                        valsOut.operator->(), count, sbit, ebit,
                                        (cudaStream_t)context.streamSpare(streamid));
        void *d_tmp = context.borrow(temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(d_tmp, temp_storage_bytes, keysIn.operator->(),
                                        keysOut.operator->(), valsIn.operator->(),
                                        valsOut.operator->(), count, sbit, ebit,
                                        (cudaStream_t)context.streamSpare(streamid));
      }
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    /// radix sort
    template <class InputIt, class OutputIt>
    void radix_sort_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                         OutputIt &&d_first, int sbit, int ebit) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::context(incomingProc).eventSpare(incomingStreamid));
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, first.operator->(),
                                     d_first.operator->(), dist, sbit, ebit,
                                     (cudaStream_t)context.streamSpare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceRadixSort::SortKeys(d_tmp, temp_storage_bytes, first.operator->(),
                                     d_first.operator->(), dist, sbit, ebit,
                                     (cudaStream_t)context.streamSpare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <class InputIt, class OutputIt>
    void radix_sort(InputIt &&first, InputIt &&last, OutputIt &&d_first, int sbit = 0,
                    int ebit
                    = sizeof(typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type)
                      * 8) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      static_assert(is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::pointer,
                              typename std::iterator_traits<remove_cvref_t<OutputIt>>::pointer>,
                    "Input iterator pointer different from output iterator\'s");
      radix_sort_impl(typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{},
                      FWD(first), FWD(last), FWD(d_first), sbit, ebit);
    }

    constexpr ProcID getProcid() const noexcept { return procid; }
    constexpr StreamID getStreamid() const noexcept { return streamid; }

  protected:
    // bool do_launch(const ParallelTask &) const noexcept;
    friend struct ExecutionPolicyInterface<CudaExecutionPolicy>;
    // template <auto flagbit> friend struct CudaLibHandle<flagbit>;

    // std::size_t blockGranularity{128};
    ProcID incomingProc{0};
    StreamID incomingStreamid{0};
    StreamID streamid{0};
    ProcID procid{0};           ///< 0-th gpu
    std::size_t shmemBytes{0};  ///< amount of shared memory passed
  };

  constexpr CudaExecutionPolicy cuda_exec() noexcept { return CudaExecutionPolicy{}; }
  constexpr CudaExecutionPolicy par_exec(cuda_exec_tag) noexcept { return CudaExecutionPolicy{}; }

}  // namespace zs
