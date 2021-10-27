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

#ifndef __SYCL_IMPL_ALGORITHM_MISMATCH__
#define __SYCL_IMPL_ALGORITHM_MISMATCH__

#include <algorithm>
#include <type_traits>

// SYCL helpers header
#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/helpers/sycl_namegen.hpp>

namespace sycl {
namespace impl {

/* mismatch.
* @brief Implementation of the command group that submits a mismatch kernel.
*/

#ifdef SYCL_PSTL_USE_OLD_ALGO

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class BinaryPredicate>
std::pair<ForwardIt1, ForwardIt2> mismatch(ExecutionPolicy& exec,
                                           ForwardIt1 first1, ForwardIt1 last1,
                                           ForwardIt2 first2, ForwardIt2 last2,
                                           BinaryPredicate p) {
  cl::sycl::queue q(exec.get_queue());
  const auto size1 = sycl::helpers::distance(first1, last1);
  const auto size2 = sycl::helpers::distance(first2, last2);

  if (size1 < 1 || size2 < 1) {
    return std::make_pair(first1, first2);
  }

  const auto device = q.get_device();

  const auto length = std::min(size1, size2);
  auto ndRange = exec.calculateNdRange(length);
  const auto local = ndRange.get_local_range()[0];

  auto buf1 = sycl::helpers::make_const_buffer(first1, first1 + length);
  auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + length);

  cl::sycl::buffer<std::size_t, 1> bufR((cl::sycl::range<1>(size1)));

  // map across the input testing whether they match the predicate
  const auto eqf = [length, ndRange, &buf1, &buf2, &bufR,
                    p](cl::sycl::handler& h) {
    const auto a1 = buf1.template get_access<cl::sycl::access::mode::read>(h);
    const auto a2 = buf2.template get_access<cl::sycl::access::mode::read>(h);
    const auto aR = bufR.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        ndRange, [a1, a2, aR, length, p](cl::sycl::nd_item<1> id) {
          const auto m_id = id.get_global_id(0);

          if (m_id < length) {
            aR[m_id] = p(a1[m_id], a2[m_id]) ? length : m_id;
          }
        });
  };
  q.submit(eqf);

  auto current_length = length;
  int passes = 0;

  const auto f = [&passes, &current_length, &ndRange, local,
                  &bufR](cl::sycl::handler& h) mutable {
    const auto aR =
        bufR.template get_access<cl::sycl::access::mode::read_write>(h);
    cl::sycl::accessor<std::size_t, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        scratch(ndRange.get_local_range(), h);

    h.parallel_for<typename ExecutionPolicy::kernelName>(
        ndRange, [aR, scratch, passes, local,
            current_length](cl::sycl::nd_item<1> id) {
          auto r = ReductionStrategy<std::size_t>(local, current_length, id,
                                                  scratch);
          r.workitem_get_from(aR);
          r.combine_threads([](std::size_t x, std::size_t y) {
            return cl::sycl::min(x, y);
          });
          r.workgroup_write_to(aR);
        });
  };
  do {
    q.submit(f);
    ++passes;
    current_length = current_length / local;
    ndRange = cl::sycl::nd_range<1>{cl::sycl::range<1>(std::max(current_length, local)),
                                    ndRange.get_local_range()};
  } while (current_length > 1);
  q.wait_and_throw();
  const auto hR = bufR.get_access<cl::sycl::access::mode::read>(
      cl::sycl::range<1>{1}, cl::sycl::id<1>{0});

  const auto mismatch_id = hR[0];
  return std::make_pair(first1 + mismatch_id, first2 + mismatch_id);
}

#else

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class BinaryPredicate>
std::pair<ForwardIt1, ForwardIt2> mismatch(ExecutionPolicy& exec,
                                           ForwardIt1 first1, ForwardIt1 last1,
                                           ForwardIt2 first2, ForwardIt2 last2,
                                           BinaryPredicate p) {
  const auto size1 = sycl::helpers::distance(first1, last1);
  const auto size2 = sycl::helpers::distance(first2, last2);

  if (size1 <= 0 || size2 <= 0) {
    return std::make_pair(first1, first2);
  }

  const auto length = std::min(size1, size2);

  const auto q = exec.get_queue();

  const auto device = q.get_device();
  using value_type1 = typename std::iterator_traits<ForwardIt1>::value_type;
  using value_type2 = typename std::iterator_traits<ForwardIt2>::value_type;

  const auto d =
      compute_mapreduce_descriptor(device, length, sizeof(value_type1));

  const auto input_buff1 =
      sycl::helpers::make_const_buffer(first1, first1 + length);
  const auto input_buff2 =
      sycl::helpers::make_const_buffer(first2, first2 + length);

  const auto pos = buffer_map2reduce(
      exec, q, input_buff1, input_buff2, length, d,
      [p, length](std::size_t pos, value_type1 x, value_type2 y) {
        return p(x, y) ? length : pos;
      },
      [](const std::size_t x, const std::size_t y) {
        return cl::sycl::min(x, y);
      });

  return std::make_pair(std::next(first1, pos), std::next(first2, pos));
}

#endif

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_MISMATCH__
