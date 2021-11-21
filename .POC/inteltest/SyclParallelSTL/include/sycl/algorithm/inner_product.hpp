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

#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_INNER_PRODUCT__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_INNER_PRODUCT__

#include <type_traits>
#include <algorithm>

#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/algorithm/buffer_algorithms.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/helpers/sycl_iterator.hpp>

namespace sycl {
namespace impl {

template <bool UseSycl>
struct InnerProductImpl;

template <>
struct InnerProductImpl<true> {
  template <class ExecutionPolicy, class InputIt1, class InputIt2, class T,
            class BinaryOperation1, class BinaryOperation2>
  static T inner_product_sequential(ExecutionPolicy &exec, InputIt1 first1,
                                    InputIt1 last1, InputIt2 first2, T value,
                                    BinaryOperation1 op1, BinaryOperation2 op2) {
    auto size = sycl::helpers::distance(first1, last1);
    if (size <= 0)
      return value;

    InputIt2 last2 = std::next(first2, size);
    auto input_buff1 = sycl::helpers::make_const_buffer(first1, last1);
    auto input_buff2 = sycl::helpers::make_const_buffer(first2, last2);

    return inner_product_sequential_sycl<typename ExecutionPolicy::kernelName>(exec.get_queue(), input_buff1,
                                                                               input_buff2, value, size, op1, op2);
  }
};

template <>
struct InnerProductImpl<false> {
  template <class ExecutionPolicy, class InputIt1, class InputIt2, class T,
            class BinaryOperation1, class BinaryOperation2>
  static T inner_product_sequential(ExecutionPolicy& exec, InputIt1 first1,
                                    InputIt1 last1, InputIt2 first2, T value,
                                    BinaryOperation1 op1, BinaryOperation2 op2) {
    while (first1 != last1) {
      value = op1(value, op2(*first1, *first2));
      ++first1;
      ++first2;
    }
    return value;
  }
};

/* sequential inner_product.
* @brief Returns the inner product of two vectors across the range [first1,
* last1) by applying Functions op1 and op2.
* Implementation of the command group that submits a for_each_n kernel,
* According to Parallelism TS version n4507. Section 4.3.2
* The kernel is implemented as a lambda.
* @param ExecutionPolicy exec : The execution policy to be used
* @param InputIterator first1 : Start of the range via a forward iterator
* @param InputIterator last1 : End of the range via a forward iterator
* @param InputIterator first2 : Start of the second vector via a forward iterator
* @param Function  op1 : No restrictions
* @param Function  op2 : No restrictions
*/
template <class ExecutionPolicy, class InputIt1, class InputIt2, class T,
          class BinaryOperation1, class BinaryOperation2>
T inner_product_sequential(ExecutionPolicy &exec, InputIt1 first1,
                           InputIt1 last1, InputIt2 first2, T value,
                           BinaryOperation1 op1, BinaryOperation2 op2) {
  static constexpr bool UseSycl = std::is_base_of<sycl::helpers::SyclIterator, InputIt1>::value &&
                                  std::is_base_of<sycl::helpers::SyclIterator, InputIt2>::value;
  return InnerProductImpl<UseSycl>::inner_product_sequential(exec, first1, last1, first2, value, op1, op2);
}

#ifdef SYCL_PSTL_USE_OLD_ALGO

/* inner_product.
* @brief Returns the inner product of two vectors across the range [first1,
* last1) by applying Functions op1 and op2. Implementation of the command group
* that submits an inner_product kernel.
*/
template <class ExecutionPolicy, class InputIt1, class InputIt2, class T,
          class BinaryOperation1, class BinaryOperation2>
T inner_product(ExecutionPolicy &exec, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value, BinaryOperation1 op1,
                BinaryOperation2 op2) {
  cl::sycl::queue q(exec.get_queue());

  auto vectorSize = sycl::helpers::distance(first1, last1);
  if (vectorSize < 1) {
    return value;
  } else {
    auto device = q.get_device();

    InputIt2 last2(first2);
    std::advance(last2, vectorSize);

    auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
    auto buf2 = sycl::helpers::make_const_buffer(first2, last2);
    cl::sycl::buffer<T, 1> bufr((cl::sycl::range<1>(vectorSize)));
    auto length = vectorSize;
    auto ndRange = exec.calculateNdRange(length);
    const auto local = ndRange.get_local_range()[0];
    int passes = 0;
    auto cg = [&passes, &length, &ndRange, local, &buf1, &buf2, &bufr, op1, op2](
        cl::sycl::handler &h) mutable {
      auto a1 = buf1.template get_access<cl::sycl::access::mode::read>(h);
      auto a2 = buf2.template get_access<cl::sycl::access::mode::read>(h);
      auto aR =
          bufr.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(ndRange.get_local_range(), h);

      h.parallel_for<typename ExecutionPolicy::kernelName>(
          ndRange, [a1, a2, aR, scratch, length, local, passes, op1, op2](
                 cl::sycl::nd_item<1> id) {
            auto r = ReductionStrategy<T>(local, length, id, scratch);
            if (passes == 0) {
              r.workitem_get_from(op2, a1, a2);
            } else {
              r.workitem_get_from(aR);
            }
            r.combine_threads(op1);
            r.workgroup_write_to(aR);
          });  // end kernel
    };         // end command group
    do {
      q.submit(cg);
      passes++;
      length = length / local;
      ndRange = cl::sycl::nd_range<1>{cl::sycl::range<1>(std::max(length, local)),
                                      ndRange.get_local_range()};
    } while (length > 1);  // end do-while
    q.wait_and_throw();
    auto hb = bufr.template get_access<cl::sycl::access::mode::read>();
    return op1(value, hb[0]);
  }
}

#else

/*
 * Inner Product Algorithm
 */

template <class ExecutionPolicy, class InputIt1, class InputIt2, class T,
          class BinaryOperation1, class BinaryOperation2>
T inner_product(ExecutionPolicy &snp, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T value, BinaryOperation1 op1,
                BinaryOperation2 op2) {

  auto q = snp.get_queue();
  auto device = q.get_device();
  auto size = sycl::helpers::distance(first1, last1);
  if (size <= 0)
    return value;
  InputIt2 last2 = std::next(first2, size);

  using value_type_1 = typename std::iterator_traits<InputIt1>::value_type;
  using value_type_2 = typename std::iterator_traits<InputIt2>::value_type;


  auto d = compute_mapreduce_descriptor(
      device, size, sizeof(value_type_1)+sizeof(value_type_2));

  auto input_buff1 = sycl::helpers::make_const_buffer(first1, last1);
  auto input_buff2 = sycl::helpers::make_const_buffer(first2, last2);

  auto map = [=](size_t pos, value_type_1 x, value_type_2 y) {
    return op2(x, y);
  };

  return buffer_map2reduce(snp, q, input_buff1, input_buff2,
                           value, d, map, op1 );
}


#endif

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_INNER_PRODUCT__
