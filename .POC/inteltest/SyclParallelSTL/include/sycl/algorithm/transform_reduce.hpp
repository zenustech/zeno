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

#ifndef __SYCL_IMPL_ALGORITHM_TRANSFORM_REDUCE__
#define __SYCL_IMPL_ALGORITHM_TRANSFORM_REDUCE__

#include <type_traits>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/algorithm/algorithm_composite_patterns.hpp>

namespace sycl {
namespace impl {

/* transform_reduce.
* @brief Returns the transform_reduce of one vector across the range [first1,
* last1) by applying Functions op1 and op2. Implementation of the command
* group
* that submits a transform_reduce kernel.
*/

#ifdef SYCL_PSTL_USE_OLD_ALGO

template <class ExecutionPolicy, class InputIterator, class UnaryOperation,
          class T, class BinaryOperation>
T transform_reduce(ExecutionPolicy& exec, InputIterator first,
                   InputIterator last, UnaryOperation unary_op, T init,
                   BinaryOperation binary_op) {
  cl::sycl::queue q(exec.get_queue());
  auto vectorSize = sycl::helpers::distance(first, last);
  
  if (vectorSize < 1) {
    return init;
  }

  cl::sycl::buffer<T, 1> bufR((cl::sycl::range<1>(vectorSize)));

  auto device = q.get_device();
  auto bufI = sycl::helpers::make_const_buffer(first, last);
  size_t length = vectorSize;
  auto ndRange = exec.calculateNdRange(vectorSize);
  const auto local = ndRange.get_local_range()[0];
  int passes = 0;

  do {
    auto f = [passes, length, ndRange, local, &bufI, &bufR, unary_op, binary_op](
        cl::sycl::handler& h) mutable {
      auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
      auto aR = bufR.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(ndRange.get_local_range(), h);

      h.parallel_for<typename ExecutionPolicy::kernelName>(
          ndRange, [aI, aR, scratch, passes, local, length, unary_op, binary_op](
                 cl::sycl::nd_item<1> id) {
            auto r = ReductionStrategy<T>(local, length, id, scratch);
            if (passes == 0) {
              r.workitem_get_from(unary_op, aI);
            } else {
              r.workitem_get_from(aR);
            }
            r.combine_threads(binary_op);
            r.workgroup_write_to(aR);
          });
    };
    q.submit(f);
    passes++;
    length = length / local;
    ndRange = cl::sycl::nd_range<1>{cl::sycl::range<1>(std::max(length, local)),
                                    ndRange.get_local_range()};
  } while (length > 1);
  q.wait_and_throw();
  auto hR = bufR.template get_access<cl::sycl::access::mode::read>();
  return binary_op(hR[0], init);
}

#else

template <typename ExecutionPolicy, typename InputIt, typename UnaryOperation,
          typename T, typename BinaryOperation>
T transform_reduce(ExecutionPolicy& snp, InputIt b,
                   InputIt e, UnaryOperation unary_op, T init,
                   BinaryOperation binary_op) {

  auto size = sycl::helpers::distance(b, e);
  if (size <= 0)
    return init;

  auto q = snp.get_queue();

  auto device = q.get_device();
  using value_type = typename std::iterator_traits<InputIt>::value_type;


  auto d = compute_mapreduce_descriptor(device, size, sizeof(value_type));

  auto input_buff = sycl::helpers::make_const_buffer(b, e);

  auto map = [=](size_t pos, value_type x) { return unary_op(x); };


  return buffer_mapreduce( snp, q, input_buff, init, d, map, binary_op );

}

#endif

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_TRANSFORM_REDUCE__
