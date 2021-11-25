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

#ifndef __SYCL_IMPL_ALGORITHM_REDUCE__
#define __SYCL_IMPL_ALGORITHM_REDUCE__

#include <type_traits>
#include <typeinfo>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/algorithm/buffer_algorithms.hpp>
#include <sycl/execution_policy>

namespace sycl {
namespace impl {

/* reduce.
 * Implementation of the command group that submits a reduce kernel.
 * The kernel is implemented as a lambda.
 * Note that there is a potential race condition while using the same buffer for
 * input-output
 */
#ifdef SYCL_PSTL_USE_OLD_ALGO
template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          typename BinaryOperation>
typename std::iterator_traits<Iterator>::value_type reduce(
    ExecutionPolicy &sep, Iterator b, Iterator e, T init, BinaryOperation bop) {
  cl::sycl::queue q(sep.get_queue());

  auto vectorSize = sycl::helpers::distance(b, e);

  if (vectorSize < 1) {
    return init;
  }

  auto device = q.get_device();

  typedef typename std::iterator_traits<Iterator>::value_type type_;
  auto bufI = sycl::helpers::make_const_buffer(b, e);
  auto length = vectorSize;
  auto ndRange = sep.calculateNdRange(length);
  const auto local = ndRange.get_local_range()[0];

  auto f = [&length, &ndRange, local, &bufI, bop](cl::sycl::handler &h) mutable {
    auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
    cl::sycl::accessor<type_, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        scratch(ndRange.get_local_range(), h);

    h.parallel_for<typename ExecutionPolicy::kernelName>(
        ndRange, [aI, scratch, local, length, bop](cl::sycl::nd_item<1> id) {
          auto r = ReductionStrategy<T>(local, length, id, scratch);
          r.workitem_get_from(aI);
          r.combine_threads(bop);
          r.workgroup_write_to(aI);
        });
  };
  do {
    q.submit(f);
    length = length / local;
    ndRange = cl::sycl::nd_range<1>{cl::sycl::range<1>(std::max(length, local)),
                                    ndRange.get_local_range()};
  } while (length > 1);
  q.wait_and_throw();
  auto hI = bufI.template get_access<cl::sycl::access::mode::read>();
  return bop(hI[0], init);
}
#else


/*
 * Reduce algorithm
 */
template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          typename BinaryOperation>
typename std::iterator_traits<Iterator>::value_type reduce(
    ExecutionPolicy &snp, Iterator b, Iterator e, T init, BinaryOperation bop) {

  auto q = snp.get_queue();
  auto device = q.get_device();
  auto size = sycl::helpers::distance(b, e);
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  if (size <= 0)
    return init;

  auto d = compute_mapreduce_descriptor(device, size, sizeof(value_type));

  auto input_buff = sycl::helpers::make_const_buffer(b, e);

  auto map = [](size_t, value_type x) { return x; };

  return buffer_mapreduce(snp, q, input_buff, init, d, map, bop);
}

#endif // __COMPUTECPP__

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_REDUCE__
