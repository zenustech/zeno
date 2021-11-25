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

#ifndef __SYCL_IMPL_ALGORITHM_FIND__
#define __SYCL_IMPL_ALGORITHM_FIND__

#include <algorithm>
#include <iostream>
#include <iterator>
#include <type_traits>

// SYCL helpers header
#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/algorithm/buffer_algorithms.hpp>
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_namegen.hpp>

namespace sycl {
namespace impl {

#ifdef SYCL_PSTL_USE_OLD_ALGO

// Implementation of a generic find algorithn to be used for implementing
// the various interfaces specified by the stl
template <class ExecutionPolicy, class InputIt, class UnaryPredicate>
InputIt find_impl(ExecutionPolicy &sep, InputIt b, InputIt e,
                  UnaryPredicate p) {
  cl::sycl::queue q(sep.get_queue());

  const auto device = q.get_device();

  // make a buffer that doesn't trigger a copy back, as we don't modify it
  auto buf = sycl::helpers::make_const_buffer(b, e);

  const auto vectorSize = buf.get_count();

  // construct a buffer to store the result of the predicate mapping stage
  auto t_buf = sycl::helpers::make_temp_buffer<std::size_t>(vectorSize);

  if (vectorSize < 1) {
    return e;
  }

  auto ndRange = sep.calculateNdRange(vectorSize);
  const auto local = ndRange.get_local_range()[0];

  // map across the input testing whether they match the predicate
  // store the result of the predicate and the index in the array of the result
  const auto eqf = [vectorSize, ndRange, &buf, &t_buf,
                    p](cl::sycl::handler &h) {
    const auto aI = buf.template get_access<cl::sycl::access::mode::read>(h);
    const auto aO = t_buf.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        ndRange, [aI, aO, vectorSize, p](cl::sycl::nd_item<1> id) {
          const auto m_id = id.get_global_id(0);
          // store index or the vector length, so that we can find the
          // _first_ index which is true, as opposed to just "one" of them
          if (m_id < vectorSize) {
            aO[m_id] = p(aI[m_id]) ? m_id : vectorSize;
          }
        });
  };
  q.submit(eqf);

  // more or less copied from the reduction implementation
  // TODO: refactor out the implementation details from both into a separate
  // module
  auto length = vectorSize;
  do {
    const auto rf = [length, ndRange, local,
                     &t_buf](cl::sycl::handler &h) {
      const auto aI =
          t_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<std::size_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(ndRange.get_local_range(), h);

      h.parallel_for<
          cl::sycl::helpers::NameGen<1, typename ExecutionPolicy::kernelName> >(
          ndRange, [aI, scratch, local, length](cl::sycl::nd_item<1> id) {
            auto r =
                ReductionStrategy<std::size_t>(local, length, id, scratch);
            r.workitem_get_from(aI);
            r.combine_threads([](std::size_t val1, std::size_t val2) {
              return cl::sycl::min(val1, val2);
            });
            r.workgroup_write_to(aI);
          });
    };
    q.submit(rf);
    length = length / local;
    ndRange = cl::sycl::nd_range<1>{cl::sycl::range<1>(std::max(length, local)),
                                    ndRange.get_local_range()};
  } while (length > 1);
  q.wait_and_throw();

  const auto hI =
      t_buf.template get_access<cl::sycl::access::mode::read>();

  // there's probably a cleaner way to do this, but essentially once we have
  // the "search index", we need to increment the begin iterator until
  // it reaches that point - we use std::advance, as not all iterators support +
  const auto search_index = hI[0];
  auto r_iter = b;
  std::advance(r_iter, search_index);
  return r_iter;
}

#else

template <typename ExecutionPolicy, typename InputIt, typename UnaryPredicate>
InputIt find_impl(ExecutionPolicy &snp, InputIt b, InputIt e,
                  UnaryPredicate p) {
  const auto size = sycl::helpers::distance(b, e);
  if (size <= 0) {
    return e;
  }

  const auto q = snp.get_queue();
  const auto device = q.get_device();
  using value_type = typename std::iterator_traits<InputIt>::value_type;

  const auto d =
      compute_mapreduce_descriptor(device, size, sizeof(std::size_t));

  const auto input_buff = sycl::helpers::make_const_buffer(b, e);

  const auto pos = buffer_mapreduce(
      snp, q, input_buff, size, d,
      [p, size](std::size_t pos, value_type x) { return p(x) ? pos : size; },
      [](std::size_t x, std::size_t y) { return cl::sycl::min(x, y); });

  if (pos == size) {
    return e;
  } else {
    return std::next(b, pos);
  }
}
#endif
}
}

#endif
