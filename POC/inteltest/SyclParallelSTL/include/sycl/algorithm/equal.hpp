/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#ifndef __SYCL_IMPL_ALGORITHM_EQUAL__
#define __SYCL_IMPL_ALGORITHM_EQUAL__

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>

// SYCL helpers header
#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/algorithm/buffer_algorithms.hpp>
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>

namespace sycl {
namespace impl {

#ifdef SYCL_PSTL_USE_OLD_ALGO

/* equal.
* @brief Implementation of the command group that submits an equal kernel.
*/
template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class BinaryPredicate>
bool equal(ExecutionPolicy& exec, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2, BinaryPredicate p) {
  cl::sycl::queue q(exec.get_queue());

  auto size1 = sycl::helpers::distance(first1, last1);
  auto size2 = sycl::helpers::distance(first2, last2);

  if (size1 != size2) {
    return false;
  }

  if (size1 < 1) {
    return true;
  }

  auto device = q.get_device();

  auto length = size1;
  auto ndRange = exec.calculateNdRange(size1);
  const auto local = ndRange.get_local_range()[0];

  auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
  auto buf2 = sycl::helpers::make_const_buffer(first2, last2);
  auto bufR = cl::sycl::buffer<bool, 1>(cl::sycl::range<1>(size1));

  do {
    int passes = 0;

    auto f = [passes, length, ndRange, local, &buf1, &buf2, &bufR,
              p](cl::sycl::handler& h) mutable {
      auto a1 = buf1.template get_access<cl::sycl::access::mode::read>(h);
      auto a2 = buf2.template get_access<cl::sycl::access::mode::read>(h);
      auto aR = bufR.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<bool, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(ndRange.get_local_range(), h);

      h.parallel_for<typename ExecutionPolicy::kernelName>(
          ndRange, [a1, a2, aR, scratch, passes, local, length,
              p](cl::sycl::nd_item<1> id) {
            auto r =
                ReductionStrategy<bool>(local, length, id, scratch);
            if (passes == 0) {
              r.workitem_get_from(p, a1, a2);
            } else {
              r.workitem_get_from(aR);
            }
            r.combine_threads(std::logical_and<bool>{});
            r.workgroup_write_to(aR);
          });  // end kernel
    };         // end command group

    q.submit(f);
    length = length / local;
    ndRange = cl::sycl::nd_range<1>{cl::sycl::range<1>(std::max(length, local)),
                                    ndRange.get_local_range()};
    ++passes;
  } while (length > 1);
  q.wait_and_throw();
  auto hr = bufR.template get_access<cl::sycl::access::mode::read>(
      cl::sycl::range<1>{1}, cl::sycl::id<1>{0});
  return hr[0];
}

#else

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class BinaryPredicate>
bool equal(ExecutionPolicy&& exec, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2, BinaryPredicate p) {
  auto q = exec.get_queue();
  auto size1 = sycl::helpers::distance(first1, last1);
  auto size2 = sycl::helpers::distance(first2, last2);

  if (size1 != size2) {
    return false;
  }

  if (size1 < 1) {
    return true;
  }

  auto device = q.get_device();
  using value_type1 = typename std::iterator_traits<ForwardIt1>::value_type;
  using value_type2 = typename std::iterator_traits<ForwardIt2>::value_type;

  auto d = compute_mapreduce_descriptor(device, size1, sizeof(std::size_t));

  auto input_buff1 = sycl::helpers::make_const_buffer(first1, last1);
  auto input_buff2 = sycl::helpers::make_const_buffer(first2, last2);

  return buffer_map2reduce(
      exec, q, input_buff1, input_buff2, true, d,
      [p](std::size_t pos, value_type1 x, value_type2 y) { return p(x, y); },
      std::logical_and<bool>{});
}

#endif

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_EQUAL_IF__
