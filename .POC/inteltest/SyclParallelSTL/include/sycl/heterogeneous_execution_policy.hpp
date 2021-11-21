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

#ifndef __SYCL_HETEROGENEOUS_EXECUTION_POLICY__
#define __SYCL_HETEROGENEOUS_EXECUTION_POLICY__

#include <cstdlib>

#include <CL/sycl.hpp>
#include <sycl/execution_policy>
#include <sycl/algorithm/transform.hpp>

namespace sycl {

/** class sycl_heterogeneous_execution_policy.
* @brief Distributes a given workload across two SYCL devices.
* It takes a float number within the range [0, 1] and it split the workload in
* two parts (chunk1 = ratio * workload.size() and chunk2 = remaining_workload),
* then it submits a command group to each device and waits for their completion.
*/
template <class KernelName>
class sycl_heterogeneous_execution_policy
    : public sycl_execution_policy<KernelName> {
  cl::sycl::queue q2;
  float ratio;

 public:
  sycl_heterogeneous_execution_policy(cl::sycl::queue q1_, cl::sycl::queue q2_,
                                      float ratio_)
      : sycl_execution_policy<KernelName>(q1_) {
    q2 = q2_;
    ratio = ratio_;
  }

  /* transform.
  * @brief Applies a Binary Operator across the range [first1, last1).
  * Implementation of the command group that submits a transform kernel,
  * According to Parallelism TS
  */
  template <class InputIt1, class InputIt2, class OutputIt,
            class BinaryOperation>
  OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                     OutputIt result, BinaryOperation binary_op) {
    auto named_sep = getNamedPolicy(*this, binary_op);

    cl::sycl::queue q1 = this->get_queue();
    int elements = std::distance(first1, last1);
    int crosspoint = elements * ratio;
    auto intermediate1(first1 + crosspoint);
    auto intermediate2(first2 + crosspoint);
    auto intermediateR(result + crosspoint);
    int q1_elems = std::distance(first1, intermediate1);
    int q2_elems = std::distance(intermediate1, last1);

    if (q1_elems < 1) {
      auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
      auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + elements);
      auto res = sycl::helpers::make_buffer(result, result + elements);
      impl::transform(named_sep, q2, buf1, buf2, res, binary_op);
      // wait for queues
      q2.wait_and_throw();
    } else if (q2_elems < 1) {
      auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
      auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + elements);
      auto res = sycl::helpers::make_buffer(result, result + elements);
      impl::transform(named_sep, q1, buf1, buf2, res, binary_op);
      // wait for queues
      q1.wait_and_throw();
    } else {
      auto buf1_q1 =
          sycl::helpers::make_const_buffer(first1, first1 + crosspoint);
      auto buf2_q1 =
          sycl::helpers::make_const_buffer(first2, first2 + crosspoint);
      auto res_q1 = sycl::helpers::make_buffer(result, result + crosspoint);
      auto buf1_q2 =
          sycl::helpers::make_const_buffer(first1 + crosspoint, last1);
      auto buf2_q2 = sycl::helpers::make_const_buffer(first2 + crosspoint,
                                                      first2 + elements);
      auto res_q2 =
          sycl::helpers::make_buffer(result + crosspoint, result + elements);
      impl::transform(named_sep, q1, buf1_q1, buf2_q1, res_q1, binary_op);
      impl::transform(named_sep, q2, buf1_q2, buf2_q2, res_q2, binary_op);
      // wait for queues
      q1.wait_and_throw();
      q2.wait_and_throw();
    }
    return last1;
  }
};

}  // sycl

#endif  // __SYCL_HETEROGENEOUS_EXECUTION_POLICY__
