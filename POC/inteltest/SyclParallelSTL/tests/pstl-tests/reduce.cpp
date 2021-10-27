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

class ReduceAlgorithm : public testing::Test {
 public:
};

TEST_F(ReduceAlgorithm, TestSyclReduce) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};
  int result = 22;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class ReduceAlgorithm> snp(q);
  int res = reduce(snp, v.begin(), v.end());

  EXPECT_EQ(res, result);
}

TEST_F(ReduceAlgorithm, TestSyclReduce2) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};
  std::vector<int> result = {32, 30, 29, 26, 21, 18, 14, 13};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class Reduce2Algorithm> snp(q);
  auto res = reduce(snp, v.begin(), v.end(), 10);

  EXPECT_EQ(res, result[0]);
}

TEST_F(ReduceAlgorithm, TestSyclReduce3) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};
  std::vector<int> result = {32, 30, 29, 26, 21, 18, 14, 13};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class Reduce3Algorithm> snp(q);
  auto res = reduce(snp, v.begin(), v.end(), 10);

  EXPECT_EQ(res, result[0]);
}

TEST_F(ReduceAlgorithm, TestSyclReduce4) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};
  std::vector<int> result = {32, 30, 29, 26, 21, 18, 14, 13};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class Reduce4Algorithm> snp(q);
  auto res = reduce(snp, v.begin(), v.end(), 10,
                    [=](int v1, int v2) { return v1 + v2; });

  EXPECT_EQ(res, result[0]);
}

TEST_F(ReduceAlgorithm, TestSyclReduce5) {
  std::vector<int> v;
  int n = 256;

  for (int i = 0; i < n; i++) {
    int x = 10 * (((float)std::rand()) / RAND_MAX);
    v.push_back(x);
  }

  int resstd = std::accumulate(v.begin(), v.end(), 0);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class Reduce5Algorithm> snp(q);
  int ressycl = reduce(snp, v.begin(), v.end());

  EXPECT_EQ(resstd, ressycl);
}

TEST_F(ReduceAlgorithm, TestSyclReduce6) {
  std::vector<int> v;
  int n = 128;

  for (int i = 0; i < n; i++) {
    int x = 10 * (((float)std::rand()) / RAND_MAX);
    v.push_back(x);
  }

  int resstd = std::accumulate(v.begin(), v.end(), 0);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class Reduce6Algorithm> snp(q);
  int ressycl = reduce(snp, v.begin(), v.end());

  EXPECT_EQ(resstd, ressycl);
}

TEST_F(ReduceAlgorithm, TestSyclReduce7) {
  std::vector<int> v;
  int n = 128;

  for (int i = 0; i < n; i++) {
    int x = 10 * (((float)std::rand()) / RAND_MAX);
    v.push_back(x);
  }

  int resstd = std::accumulate(v.begin(), v.end(), 10);

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class Reduce7Algorithm> snp(q);
  int ressycl = reduce(snp, v.begin(), v.end(), 10,
                       [=](int val1, int val2) { return val1 + val2; });

  EXPECT_EQ(resstd, ressycl);
}
