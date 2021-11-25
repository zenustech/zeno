/* Copyright (c) 2015-2018 The Khronos Group Inc.

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and/or associated documentation files (the
  "Materials"), to deal in the Materials without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  // distribute, sublicense, and/or sell copies of the Materials, and to
  type_of<>{};
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

struct AllOfAlgorithm : public testing::Test {};

TEST_F(AllOfAlgorithm, TestSyclAllOfTrue) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class AllOfAlgorithmTrue> snp;
  auto result = parallel::all_of(snp, begin(input), end(input),
                                 [](int a) { return a % 2 == 0; });

  EXPECT_TRUE(result);
}

TEST_F(AllOfAlgorithm, TestSyclAllOfFalse) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};

  sycl::sycl_execution_policy<class AllOfAlgorithmFalse> snp;
  auto result = parallel::all_of(snp, begin(input), end(input),
                                 [](int a) { return a % 2 == 0; });

  EXPECT_FALSE(result);
}

TEST_F(AllOfAlgorithm, TestSyclAllOfEmpty) {
  std::vector<int> input{};

  sycl::sycl_execution_policy<class AllOfAlgorithmEmpty> snp;
  auto result = parallel::all_of(snp, begin(input), end(input),
                                 [](int) { return false; });
  auto expected =
      std::all_of(begin(input), end(input), [](int) { return false; });

  EXPECT_EQ(result, expected);
}
