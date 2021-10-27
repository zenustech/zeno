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

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

struct TransformAlgorithm : public testing::Test {};

TEST_F(TransformAlgorithm, TestStdTransform) {
  std::vector<int> v = {2, 1, 3};
  std::vector<int> result = {3, 2, 4};

  std::transform(v.begin(), v.end(), v.begin(),
                 [=](int val) { return val + 1; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), result.begin()));
}

TEST_F(TransformAlgorithm, TestSyclTransform) {
  std::vector<int> v = {2, 1, 3};
  std::vector<int> o = {2, 1, 3};
  std::vector<int> result = {3, 2, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TransformAlgorithm> snp(q);
  transform(snp, v.begin(), v.end(), v.begin(),
            [=](int val) { return val - 1; });

  sycl::sycl_execution_policy<class TransformAlgorithm2> snp2(q);
  transform(snp2, v.begin(), v.end(), o.begin(),
            [=](int val) { return val + 2; });
  EXPECT_TRUE(std::equal(o.begin(), o.end(), result.begin()));
}

TEST_F(TransformAlgorithm, TestSycl2Transform) {
  std::vector<int> v1 = {2, 1, 3, 5, 6};
  std::vector<int> v2 = {2, 1, 3, 3, 1};
  std::vector<int> o = {0, 0, 0, 0, 0};
  std::vector<int> result = {5, 3, 7, 9, 8};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TransformAlgorithm3> snp(q);

  transform(snp, v1.begin(), v1.end(), v2.begin(), o.begin(),
            [=](int val1, int val2) { return val1 + val2 + 1; });
  EXPECT_TRUE(std::equal(o.begin(), o.end(), result.begin()));
}

TEST_F(TransformAlgorithm, TestSycl4Transform) {
  std::vector<int> v;
  std::vector<int> res_std;
  std::vector<int> res_sycl;
  int n = 4096;

  for (int i = 0; i < n; i++) {
    int x = 10 * (((float)std::rand()) / RAND_MAX);
    v.push_back(x);
    res_std.push_back(0);
    res_sycl.push_back(0);
  }

  std::transform(v.begin(), v.end(), res_std.begin(),
                 [=](int val) { return val + 1; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TransformAlgorithm4> snp(q);
  transform(snp, v.begin(), v.end(), res_sycl.begin(),
            [=](int val1) { return val1 + 1; });
  EXPECT_TRUE(std::equal(res_std.begin(), res_std.end(), res_sycl.begin()));
}
