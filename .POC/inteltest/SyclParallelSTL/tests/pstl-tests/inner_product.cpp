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
#include <list>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

struct InnerProductAlgorithm : public testing::Test {};

TEST_F(InnerProductAlgorithm, TestStdInnerProduct) {
  std::vector<int> v1 = {2, 2, 3, 1};
  std::vector<int> v2 = {4, 2, 1, 3};
  int result = 18;
  int value = 0;

  value = std::inner_product(v1.begin(), v1.end(), v2.begin(), value);

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSyclInnerProduct) {
  std::vector<int> v1 = {2, 2, 3, 1};
  std::vector<int> v2 = {4, 2, 1, 3};
  int result = 28;
  int value = 10;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCLInnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value);

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSycl2InnerProduct) {
  std::vector<int> v1 = {2, 2, 3, 1, 5, 1, 1, 1};
  std::vector<int> v2 = {4, 2, 1, 3, 5, 1, 1, 1};
  int result = 46;
  int value = 0;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL2InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value);

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSycl3InnerProduct) {
  std::vector<int> v1 = {2, 2, 3, 1, 5, 1, 1};
  std::vector<int> v2 = {4, 2, 1, 3, 5, 1, 1};
  int result = 45;
  int value = 0;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL3InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value);

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSycl4InnerProduct) {
  std::vector<int> v1 = {2, 2, 3, 1};
  std::vector<int> v2 = {4, 2, 1, 3};
  int result = 28;
  int value = 10;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL4InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value,
                        [=](int v1, int v2) { return v1 + v2; },
                        [=](int v1, int v2) { return v1 * v2; });

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSycl5InnerProduct) {
  std::vector<float> v1 = {2.0, 2.0, 3.0, 1.0};
  std::vector<float> v2 = {4.0, 2.0, 1.0, 3.0};
  float result = 28.0;
  float value = 10.0;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL5InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value,
                        [=](float v1, float v2) { return v1 + v2; },
                        [=](float v1, float v2) { return v1 * v2; });

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSycl6InnerProduct) {
  std::list<float> v1;  //{2.0, 2.0, 3.0, 1.0};
  std::list<float> v2;  //{4.0, 2.0, 1.0, 3.0};
  float result = 28.0;
  float value = 10.0;

  v1.push_back(2.0);
  v1.push_back(2.0);
  v1.push_back(3.0);
  v1.push_back(1.0);

  v2.push_back(4.0);
  v2.push_back(2.0);
  v2.push_back(1.0);
  v2.push_back(3.0);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL6InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value,
                        [=](float v1, float v2) { return v1 + v2; },
                        [=](float v1, float v2) { return v1 * v2; });

  EXPECT_TRUE(result == value);
}

TEST_F(InnerProductAlgorithm, TestSycl7InnerProduct) {
  std::vector<int> v1;
  std::vector<int> v2;
  int n_elems = 128;
  for (int i = 0; i < n_elems; i++) {
    v1.push_back(1);
    v2.push_back(2);
  }
  int value = 0;
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL7InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value,
                        [=](int v1, int v2) { return v1 + v2; },
                        [=](int v1, int v2) { return v1 * v2; });

  EXPECT_TRUE( (128*2) == value);
}

TEST_F(InnerProductAlgorithm, TestSycl8InnerProduct) {
  std::vector<int> v1;
  std::vector<int> v2;
  int n_elems = 128;
  for (int i = 0; i < n_elems; i++) {
    v1.push_back(1);
    v2.push_back(2);
  }
  int value = 0;
  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SYCL8InnerProductAlgorithm> snp(q);
  value = inner_product(snp, v1.begin(), v1.end(), v2.begin(), value);

  EXPECT_TRUE( (128*2) == value);
}
