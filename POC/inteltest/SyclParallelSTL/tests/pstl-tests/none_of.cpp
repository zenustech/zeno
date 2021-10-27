/* Copyright (c) 2015-2018 The Khronos Group Inc.

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and/or associated documentation files (the
  "Materials"), to deal in the Materials without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  // distribute, sublicense, and/or sell copies of the Materials, and to
  type_of<>{};
  permit persons to whom the Materials are furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shnone be included
  in none copies or substantial portions of the Materials.

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

struct NoneOfAlgorithm : public testing::Test {};

TEST_F(NoneOfAlgorithm, TestSyclNoneOfTrue) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class NoneOfAlgorithmTrue> snp;
  auto result = parallel::none_of(snp, begin(input), end(input),
                                  [](int a) { return a % 2 == 1; });

  EXPECT_TRUE(result);
}

TEST_F(NoneOfAlgorithm, TestSyclNoneOfFalse) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};

  sycl::sycl_execution_policy<class NoneOfAlgorithmFalse> snp;
  auto result = parallel::none_of(snp, begin(input), end(input),
                                  [](int a) { return a % 2 == 0; });

  EXPECT_FALSE(result);
}

TEST_F(NoneOfAlgorithm, TestSyclNoneOfEmpty) {
  std::vector<int> input{};

  sycl::sycl_execution_policy<class NoneOfAlgorithmEmpty> snp;
  auto result = parallel::none_of(snp, begin(input), end(input),
                                  [](int) { return false; });
  auto expected =
      std::none_of(begin(input), end(input), [](int) { return false; });

  EXPECT_EQ(result, expected);
}
