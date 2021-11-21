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
#include <numeric>

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

struct InclusiveScanAlgorithm : public testing::Test {};

/*
 * we define functors plus and multiplies as there is not enough guarentee on
 * their implementation in the standard library.
 * In particular it is not guarenteed that std::plus and std::multiplies have
 * no side effects
 */

template<typename T>
struct plus {
  constexpr T operator() (const T &x, const T &y) const {
    return x + y;
  }
};

template<typename T>
struct multiplies {
  constexpr T operator() (const T &x, const T &y) const {
    return x * y;
  }
};

template <typename T, class BinaryOperation>
void inclusive_scan_gold(std::vector<T>& v, T init = 0,
                         BinaryOperation bop = plus<T>()) {
  v[0] = bop(v[0], init);
  for (auto i = std::next(v.begin()); i != v.end(); i++) {
    *i = bop(*i, *(std::prev(i)));
  }
}

// test the gold computation against a known value
TEST_F(InclusiveScanAlgorithm, TestSTDInclusiveScan) {
  std::vector<int> v    = {5, 1,  6,  2,  6,  2,  5,  7};
  std::vector<int> gold = {5, 6, 12, 14, 20, 22, 27, 34};

  inclusive_scan_gold(v, 0, plus<int>());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// tests with small power of two vectors
// 1: default addition + operation
// 2: custom addition operation
// 3: custom addition and initial value

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanPowerOfTwo1) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 0, plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmPOT1> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanPowerOfTwo2) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 0, plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmPOT2> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 plus<int>());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanPowerOfTwo3) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 10, plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmPOT3> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 plus<int>(), 10);

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// Tests with power of two input, non additive operation, and default/other init
// 1: Default initial value
// 2: Special default value

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanMultOperation1) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 0, multiplies<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmMOP1> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 multiplies<int>());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanMultOperation2) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 42, multiplies<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmMOP2> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 multiplies<int>(), 42);

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// tests with small _non_ power of two vectors
// 1: default addition + operation
// 2: custom addition operation
// 3: custom addition and initial value

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanNonPowerOfTwo1) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 0, std::plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmNPOT1> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanNonPowerOfTwo2) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 0, std::plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmNPOT2> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 plus<int>());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanNonPowerOfTwo3) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5};
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 10, std::plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmNPOT3> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 plus<int>(), 10);

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// test of a large power of two sized input
TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanLargePowerOfTwo) {
  std::vector<int> v(1 << 12);
  std::fill(v.begin(), v.end(), 42);
  std::vector<int> gold(v);

  inclusive_scan_gold(gold, 0, std::plus<int>());

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class InclusiveScanAlgorithmLPOT> snp(q);

  inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                 plus<int>());

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// test of a variety of non power of two sized input
TEST_F(InclusiveScanAlgorithm, TestSyclInclusiveScanNonPowerOfTwoRange) {
  for (int i = 5; i < 14; i++) {
    std::vector<int> v((1 << i) - 1);
    std::fill(v.begin(), v.end(), 42);
    std::vector<int> gold(v);

    inclusive_scan_gold(gold, 0, std::plus<int>());

    cl::sycl::queue q;
    sycl::sycl_execution_policy<class InclusiveScanAlgorithmNPOTR> snp(q);

    inclusive_scan(snp, v.begin(), v.end(), v.begin(),
                   plus<int>());

    EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
  }
}
