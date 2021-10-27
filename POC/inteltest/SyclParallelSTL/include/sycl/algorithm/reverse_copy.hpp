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

#ifndef __SYCL_IMPL_ALGORITHM_REVERSE_COPY__
#define __SYCL_IMPL_ALGORITHM_REVERSE_COPY__

#include <algorithm>
#include <iostream>
#include <type_traits>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

/* reverse_copy.
 * Implementation of the command group that submits a reverse_copy kernel.
 * The kernel is implemented as a lambda.
 */
template <class ExecutionPolicy, class BidirIt, class ForwardIt>
ForwardIt reverse_copy(ExecutionPolicy &sep, BidirIt first, BidirIt last,
                       ForwardIt d_first) {
  cl::sycl::queue q{sep.get_queue()};
  const auto device = q.get_device();

  auto bufI = helpers::make_buffer(first, last);
  const auto d_last(d_first + bufI.get_count());
  auto bufO = sycl::helpers::make_buffer(d_first, d_last);

  const auto vectorSize = bufI.get_count();
  const auto ndRange = sep.calculateNdRange(vectorSize);
  const auto f = [vectorSize, ndRange, &bufI,
            &bufO](cl::sycl::handler &h) mutable {
    const auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
    const auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<typename ExecutionPolicy::kernelName>(
        ndRange, [aI, aO, vectorSize](cl::sycl::nd_item<1> id) {
          const auto global_id = id.get_global_id(0);
          if (global_id < vectorSize) {
            aO[global_id] = aI[vectorSize - global_id - 1];
          }
        });
  };
  q.submit(f);

  return d_last;
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_REVERSE_COPY__
