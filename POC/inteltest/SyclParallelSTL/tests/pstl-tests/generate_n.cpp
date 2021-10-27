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
#include <iterator>
#include <vector>

#include <experimental/algorithm>
#include <sycl/execution_policy>

namespace parallel = std::experimental::parallel;

struct GenerateNAlgorithm : public testing::Test {};

TEST_F(GenerateNAlgorithm, TestStdGenerateN) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = {1, 1, 1, 1, 1, 1, 1, 1};

  std::generate_n(begin(v), v.size(), []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}

TEST_F(GenerateNAlgorithm, TestStd2GenerateN) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = {1, 1, 1, 1, 5, 6, 7, 8};

  std::generate_n(begin(v), 4, []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}

TEST_F(GenerateNAlgorithm, TestStd3GenerateN) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = v;

  int negative_count = -v.size();
  std::generate_n(begin(v), negative_count, []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}

TEST_F(GenerateNAlgorithm, TestSyclGenerateN) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = {1, 1, 1, 1, 1, 1, 1, 1};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class GenerateNAlgorithm> snp(q);
  parallel::generate_n(snp, begin(v), v.size(), []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}

TEST_F(GenerateNAlgorithm, TestSycl2GenerateN) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = {1, 1, 1, 1, 5, 6, 7, 8};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class GenerateN2Algorithm> snp(q);
  parallel::generate_n(snp, begin(v), 4, []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}

TEST_F(GenerateNAlgorithm, TestSycl3GenerateN) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = v;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class GenerateN3Algorithm> snp(q);
  int negative_count = -v.size();
  parallel::generate_n(snp, begin(v), negative_count, []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}
