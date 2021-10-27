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

struct RotateCopyAlgorithm : public testing::Test {};

template <class ExecutionPolicy, class C1, class C2, class ForwardIt>
void test_rotate_copy(ExecutionPolicy &sep, C1 &in, C2 &out, ForwardIt middle) {
  C2 expected(out);

  std::rotate_copy(begin(in), middle, end(in), begin(expected));

  auto ret = parallel::rotate_copy(sep, begin(in), middle, end(in), begin(out));

  EXPECT_TRUE(std::equal(begin(out), ret, begin(expected)));
  EXPECT_TRUE(std::equal(begin(expected), end(expected), begin(out)));
}

TEST_F(RotateCopyAlgorithm, TestSyclRotateCopy0) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateCopyAlgorithm0> sep(q);
  std::vector<int> in(0), out(in.size());
  test_rotate_copy(sep, in, out, begin(in));
}

TEST_F(RotateCopyAlgorithm, TestSyclRotateCopy1) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateCopyAlgorithm1> sep(q);
  std::vector<int> in{42}, out(in.size());
  test_rotate_copy(sep, in, out, begin(in));
}

TEST_F(RotateCopyAlgorithm, TestSyclRotateCopy8) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateCopyAlgorithm8> sep(q);
  std::vector<int>  in{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<int,8> out;
  test_rotate_copy(sep, in, out, std::next(begin(in)));
}

TEST_F(RotateCopyAlgorithm, TestSyclRotateCopy9) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateCopyAlgorithm9> sep(q);
  std::vector<int> in{1, 2, 3, 4, 5, 6, 7, 8, 9}, out(in.size());
  test_rotate_copy(sep, in, out, std::next(begin(in),0));
  test_rotate_copy(sep, in, out, std::next(begin(in),4));
  test_rotate_copy(sep, in, out, std::next(begin(in),8));
}

TEST_F(RotateCopyAlgorithm, TestSyclRotateCopy10) {
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class RotateCopyAlgorithm10> sep(q);
  std::deque<int>        in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::forward_list<int> out(in.size());
  test_rotate_copy(sep, in, out, std::next(begin(in),6));
}
