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

struct EqualAlgorithm : public testing::Test {};

TEST_F(EqualAlgorithm, EqualTrue) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class EqualAlgorithmTrue> snp;
  auto result =
      parallel::equal(snp, begin(input), end(input), begin(input),
                      end(input), [](int a, int b) { return a == b; });

  EXPECT_TRUE(result);
}

TEST_F(EqualAlgorithm, EqualTrueJustBegin) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class EqualAlgorithmTrueJustBegin> snp;
  auto result = parallel::equal(snp, begin(input), end(input), begin(input),
                                [](int a, int b) { return a == b; });

  EXPECT_TRUE(result);
}

TEST_F(EqualAlgorithm, EqualFalse) {
  std::vector<int> input1 = {2, 4, 6, 8, 10, 12, 14, 16};
  std::vector<int> input2 = {2, 4, 6, 8, 0, 12, 14, 16};

  sycl::sycl_execution_policy<class EqualAlgorithmFalse> snp;
  auto result =
      parallel::equal(snp, begin(input1), end(input1), begin(input2),
                      end(input2), [](int a, int b) { return a == b; });

  EXPECT_FALSE(result);
}

TEST_F(EqualAlgorithm, EqualCustomPredicate) {
  std::vector<int> input1 = {2, 4, 6, 8, 10, 12, 14, 16};
  std::vector<int> input2 = {0, 0, 0, 0, 0, 0, 0, 0};

  sycl::sycl_execution_policy<class EqualAlgorithmCustomPredicate> snp;
  auto result =
      parallel::equal(snp, begin(input1), end(input1), begin(input2),
                      end(input2), [](int a, int b) { return a != b; });

  EXPECT_TRUE(result);
}

TEST_F(EqualAlgorithm, EqualNoPredicate) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class EqualAlgorithmNoPredicate> snp;
  auto result = parallel::equal(snp, begin(input), end(input), begin(input),
                                end(input));

  EXPECT_TRUE(result);
}

TEST_F(EqualAlgorithm, EqualNoPredicateJustBegin) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class EqualAlgorithmNoPredicateJustBegin> snp;
  auto result = parallel::equal(snp, begin(input), end(input), begin(input));

  EXPECT_TRUE(result);
}

TEST_F(EqualAlgorithm, EqualDifferentLength) {
  std::vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16};

  sycl::sycl_execution_policy<class EqualAlgorithmShorter> snp;
  auto result =
      parallel::equal(snp, begin(input), end(input), begin(input),
                      end(input) - 1, [](int a, int b) { return a == b; });

  EXPECT_FALSE(result);
}

TEST_F(EqualAlgorithm, EqualEmpty) {
  std::vector<int> input{};

  sycl::sycl_execution_policy<class EqualAlgorithmEmpty> snp;
  auto result =
      parallel::equal(snp, begin(input), end(input), begin(input),
                      end(input), [](int a, int b) { return a == b; });
  auto expected = std::equal(begin(input), end(input), begin(input),
                             [](int a, int b) { return a == b; });

  EXPECT_EQ(result, expected);
}

TEST_F(EqualAlgorithm, NotEqualFloat) {
  std::vector<float> input1{1.0f, 1.5f, 2.0f, 4.0f};
  std::vector<float> input2{1.0f, 1.2f, 2.0f, 4.0f};

  sycl::sycl_execution_policy<class EqualAlgorithmNotEqualFloat> snp;
  auto result =
      parallel::equal(snp, begin(input1), end(input1), begin(input2),
                      end(input2), [](int a, int b) { return a == b; });
  auto expected = std::equal(begin(input1), end(input1), begin(input2),
                             [](int a, int b) { return a == b; });

  EXPECT_EQ(result, expected);
}
