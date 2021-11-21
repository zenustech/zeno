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
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include <experimental/algorithm>
#include <sycl/execution_policy>

namespace parallel = std::experimental::parallel;

class MismatchAlgorithm : public testing::Test {
 public:
};

TEST_F(MismatchAlgorithm, TestMismatchEqual) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), end(v), begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmEqual> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v), begin(v), end(v),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchMatchCustomPredicate) {
  std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> v2{1, 2, 3, 4, 5, 6, 7, 8};

  auto expected =
      std::mismatch(begin(v1), end(v1), begin(v2), std::less<int>{});

  sycl::sycl_execution_policy<class MismatchAlgorithmMatchCustomPredicate>
      snp{};
  auto actual = parallel::mismatch(snp, begin(v1), end(v1), begin(v2), end(v2),
                                   std::less<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchNotEqualCustomPredicate) {
  std::vector<int> v1{0, 1, 2, 9, 4, 5, 6, 7};
  std::vector<int> v2{1, 2, 3, 4, 5, 6, 7, 8};

  auto expected =
      std::mismatch(begin(v1), end(v1), begin(v2), std::less<int>{});

  sycl::sycl_execution_policy<class MismatchAlgorithmNotEqualCustomPredicate>
      snp{};
  auto actual = parallel::mismatch(snp, begin(v1), end(v1), begin(v2), end(v2),
                                   std::less<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchEqualOneRange) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), end(v), begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmEqualOneRange> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v), begin(v), end(v),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchMatchOneRangeCustomPredicate) {
  std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> v2{1, 2, 3, 4, 5, 6, 7, 8};

  auto expected =
      std::mismatch(begin(v1), end(v1), begin(v2), std::less<int>{});

  sycl::sycl_execution_policy<
      class MismatchAlgorithmMatchOneRangeCustomPredicate>
      snp{};
  auto actual =
      parallel::mismatch(snp, begin(v1), end(v1), begin(v2), std::less<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchNoMatchOneRangeCustomPredicate) {
  std::vector<int> v1{0, 1, 9, 3, 4, 5, 6, 7};
  std::vector<int> v2{1, 2, 3, 4, 5, 6, 7, 8};

  auto expected =
      std::mismatch(begin(v1), end(v1), begin(v2), std::less<int>{});

  sycl::sycl_execution_policy<
      class MismatchAlgorithmNoMatchOneRangeCustomPredicate>
      snp{};
  auto actual =
      parallel::mismatch(snp, begin(v1), end(v1), begin(v2), std::less<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchMatchOneRangeNoPredicate) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), end(v), begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmMatchNoPredicate> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v), begin(v));

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchEqualNoPredicate) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), end(v), begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmEqualNoPredicate> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v), begin(v), end(v));

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchEqualFirstSmaller) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), end(v) - 2, begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmEqualFirstSmaller> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v) - 2, begin(v), end(v),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchEqualSecondSmaller) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), end(v) - 2, begin(v));
  using std::swap;
  swap(std::get<0>(expected), std::get<1>(expected));

  sycl::sycl_execution_policy<class MismatchAlgorithmEqualSecondSmaller> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v), begin(v), end(v) - 2,
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchNotEqual) {
  std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> v2{0, 1, 2, 3, 6, 5, 6, 7};

  auto expected = std::mismatch(begin(v1), end(v1), begin(v2));

  sycl::sycl_execution_policy<class MismatchAlgorithmNotEqual> snp{};
  auto actual = parallel::mismatch(snp, begin(v1), end(v1), begin(v2), end(v2),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchNotEqualNoPredicate) {
  std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> v2{0, 1, 2, 3, 6, 5, 6, 7};

  auto expected = std::mismatch(begin(v1), end(v1), begin(v2));

  sycl::sycl_execution_policy<class MismatchAlgorithmNotEqualNoPredicate> snp{};
  auto actual = parallel::mismatch(snp, begin(v1), end(v1), begin(v2), end(v2));

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchFirstEmpty) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), begin(v), begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmFirstEmpty> snp{};
  auto actual = parallel::mismatch(snp, begin(v), begin(v), begin(v), end(v),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchSecondEmpty) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), begin(v), begin(v));
  using std::swap;
  swap(std::get<0>(expected), std::get<1>(expected));

  sycl::sycl_execution_policy<class MismatchAlgorithmSecondEmpty> snp{};
  auto actual = parallel::mismatch(snp, begin(v), end(v), begin(v), begin(v),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchBothEmpty) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(begin(v), begin(v), begin(v));

  sycl::sycl_execution_policy<class MismatchAlgorithmBothEmpty> snp{};
  auto actual = parallel::mismatch(snp, begin(v), begin(v), begin(v), begin(v),
                                   std::equal_to<int>{});

  EXPECT_EQ(actual, expected);
}
