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
#include "gmock/gmock.h"

#include <algorithm>
#include <vector>
#include <deque>
#include <forward_list>
#include <iterator>

#include <experimental/algorithm>
#include <sycl/execution_policy>

namespace parallel = std::experimental::parallel;

struct RotateAlgorithm : public testing::Test {};

template <class ExecutionPolicy, class C, class ForwardIt>
void test_rotate(ExecutionPolicy &sep, C &in, ForwardIt middle) {
  C expected(in);
  auto middle_e = std::next(begin(expected),std::distance(begin(in),middle));

  auto ret_e = std::rotate(begin(expected), middle_e, end(expected));

  auto ret = parallel::rotate(sep, begin(in), middle, end(in));

  EXPECT_TRUE(std::equal(ret_e,           end(expected), ret));
  EXPECT_TRUE(std::equal(begin(expected), end(expected), begin(in)));
}

TEST_F(RotateAlgorithm, TestSyclRotate0) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateAlgorithm0> sep(q);
  std::vector<int> in(0);
  test_rotate(sep, in, begin(in));
}

TEST_F(RotateAlgorithm, TestSyclRotate1) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateAlgorithm1> sep(q);
  std::vector<int> in{42};
  test_rotate(sep, in, begin(in));
}

TEST_F(RotateAlgorithm, TestSyclRotate8) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateAlgorithm8> sep(q);
  std::array<int,8> in{1, 2, 3, 4, 5, 6, 7, 8};
  test_rotate(sep, in, std::next(begin(in)));
}

TEST_F(RotateAlgorithm, TestSyclRotate9) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateAlgorithm9> sep(q);
  std::vector<int> in{1, 2, 3, 4, 5, 6, 7, 8, 9};
  test_rotate(sep, in, std::next(begin(in),0));
  test_rotate(sep, in, std::next(begin(in),4));
  test_rotate(sep, in, std::next(begin(in),8));
}

TEST_F(RotateAlgorithm, TestSyclRotate10) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateAlgorithm10> sep(q);
  std::deque<int> in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  test_rotate(sep, in, std::next(begin(in),6));
}
