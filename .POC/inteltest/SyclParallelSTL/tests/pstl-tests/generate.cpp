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

struct GenerateAlgorithm : public testing::Test {};

TEST_F(GenerateAlgorithm, TestStdGenerate) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = {1, 1, 1, 1, 1, 1, 1, 1};

  std::generate(begin(v), end(v), []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}

TEST_F(GenerateAlgorithm, TestSyclGenerate) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result = {1, 1, 1, 1, 1, 1, 1, 1};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class GenerateAlgorithm> snp(q);
  parallel::generate(snp, begin(v), end(v), []() { return 1; });

  EXPECT_TRUE(std::equal(begin(v), end(v), begin(result)));
}
