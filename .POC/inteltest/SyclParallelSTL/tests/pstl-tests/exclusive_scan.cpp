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

struct ExclusiveScanAlgorithm : public testing::Test {};

template <typename T, class BinaryOperation>
void exclusive_scan_gold(std::vector<T>& v, T init = 0,
                         BinaryOperation bop = [](T a, T b) { return a + b; }) {
  // linear shift right:
  T tmp = *(v.begin());
  for (auto i = std::next(v.begin()); i != v.end(); i++) {
    std::swap(tmp, *i);
  }
  // set the initial element
  v[0] = init;
  // inclusive scan
  for (auto i = std::next(v.begin()); i != v.end(); i++) {
    *i = bop(*i, *(std::prev(i)));
  }
}

// test the gold computation against a known value
TEST_F(ExclusiveScanAlgorithm, TestSTDExclusiveScan) {
  std::vector<int> v    = {5, 1, 6,  2,  6,  2,  5,  7};
  std::vector<int> gold = {0, 5, 6, 12, 14, 20, 22, 27};

  exclusive_scan_gold(v, 0, [](int a, int b) { return a + b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// tests with small power of two vectors
// 1: default addition + operation
// 2: custom addition operation
// 3: custom addition and initial value

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanPowerOfTwo1) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 0, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmPOT1> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 0);

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanPowerOfTwo2) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 0, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmPOT2> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 0,
                 [](int a, int b) { return a + b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanPowerOfTwo3) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 10, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmPOT3> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 10,
                 [](int a, int b) { return a + b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// Tests with power of two input, non additive operation, and default/other init
// 1: Default initial value
// 2: Special default value

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanMultOperation1) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 1, [](int a, int b) { return a * b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmMOP1> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 1,
                 [](int a, int b) { return a * b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanMultOperation2) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5, 7};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 42, [](int a, int b) { return a * b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmMOP2> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 42,
                 [](int a, int b) { return a * b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// tests with small _non_ power of two vectors
// 1: default addition + operation
// 2: custom addition operation
// 3: custom addition and initial value

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanNonPowerOfTwo1) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 0, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmNPOT1> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 0);

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanNonPowerOfTwo2) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 0, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmNPOT2> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 0,
                 [](int a, int b) { return a + b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanNonPowerOfTwo3) {
  std::vector<int> v = {5, 1, 6, 2, 6, 2, 5};
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 10, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmNPOT3> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 10,
                 [](int a, int b) { return a + b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// test of a large power of two sized input
TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanLargePowerOfTwo) {
  std::vector<int> v(1 << 9);
  std::fill(v.begin(), v.end(), 42);
  std::vector<int> gold(v);

  exclusive_scan_gold(gold, 0, [](int a, int b) { return a + b; });

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ExclusiveScanAlgorithmLPOT> snp(q);
  exclusive_scan(snp, v.begin(), v.end(), v.begin(), 0,
                 [](int a, int b) { return a + b; });

  EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
}

// test of a variety of non power of two sized input
TEST_F(ExclusiveScanAlgorithm, TestSyclExclusiveScanNonPowerOfTwoRange) {
  for (int i = 5; i < 9; i++) {
    std::vector<int> v((1 << i) - 1);
    std::fill(v.begin(), v.end(), 42);
    std::vector<int> gold(v);

    exclusive_scan_gold(gold, 0, [](int a, int b) { return a + b; });

    cl::sycl::queue q;
    sycl::sycl_execution_policy<class ExclusiveScanAlgorithmNPOTR> snp(q);
    exclusive_scan(snp, v.begin(), v.end(), v.begin(), 0,
                   [](int a, int b) { return a + b; });

    EXPECT_TRUE(std::equal(v.begin(), v.end(), gold.begin()));
  }
}
