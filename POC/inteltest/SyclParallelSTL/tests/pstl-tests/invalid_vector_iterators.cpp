// Tests for checking what we do when passing invalid iterators to
// parallel algorithms, and making sure that we exception or error in a
// sensible way.

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

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

typedef sycl::helpers::negative_iterator_distance expected_exception;

class InvalidIterators : public testing::Test {
 public:
};

TEST_F(InvalidIterators, TestReduce1) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestReduce1> snp(q);
  ASSERT_THROW({ reduce(snp, v.end(), v.begin()); }, expected_exception);
}

TEST_F(InvalidIterators, TestReduce2) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestReduce2> snp(q);

  ASSERT_THROW({ reduce(snp, v.rend(), v.rbegin()); }, expected_exception);
}

TEST_F(InvalidIterators, TestReduce3) {
  std::vector<int> v = {2, 1, 3, 5, 3, 4, 1, 3};
  std::vector<int> result = {22};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestReduce3> snp(q);
  int res;
  ASSERT_NO_THROW({ res = reduce(snp, v.rbegin(), v.rend()); });
  EXPECT_TRUE(res == result[0]);
}

TEST_F(InvalidIterators, TestTransformReduce1) {
  std::vector<int> v = {2, 1, 3, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestTransformReduce1> snp(q);
  ASSERT_THROW({
    transform_reduce(snp, v.end(), v.begin(), [=](int val) { return val * 2; },
                     0, [=](int v1, int v2) { return v1 + v2; });
  }, expected_exception);
}

TEST_F(InvalidIterators, TestTransformReduce2) {
  std::vector<int> v = {2, 1, 3, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestTransformReduce2> snp(q);
  ASSERT_THROW({
    transform_reduce(snp, v.rend(), v.rbegin(),
                     [=](int val) { return val * 2; }, 0,
                     [=](int v1, int v2) { return v1 + v2; });
  }, expected_exception);
}

TEST_F(InvalidIterators, TestTransformReduce3) {
  std::vector<int> v = {2, 1, 3, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestTransformReduce3> snp(q);
  ASSERT_NO_THROW({
    transform_reduce(snp, v.rbegin(), v.rend(),
                     [=](int val) { return val * 2; }, 0,
                     [=](int v1, int v2) { return v1 + v2; });
  });
}

TEST_F(InvalidIterators, TestCountIf1) {
  std::vector<float> v;
  int n_elems = 128;

  for (int i = 0; i < n_elems; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    v.push_back(x);
  }

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestCountIf1> snp(q);

  ASSERT_THROW({
    count_if(snp, v.end(), v.begin(), [=](float v1) { return v1 < 0.5; });
  }, expected_exception);
}

TEST_F(InvalidIterators, TestCountIf2) {
  std::vector<float> v;
  int n_elems = 128;

  for (int i = 0; i < n_elems; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    v.push_back(x);
  }

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestCountIf2> snp(q);

  ASSERT_THROW({
    count_if(snp, v.rend(), v.rbegin(), [=](float v1) { return v1 < 0.5; });
  }, expected_exception);
}

TEST_F(InvalidIterators, TestCountIf3) {
  std::vector<float> v;
  int n_elems = 128;

  for (int i = 0; i < n_elems; i++) {
    float x = ((float)std::rand()) / RAND_MAX;
    v.push_back(x);
  }

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TestCountIf3> snp(q);

  ASSERT_NO_THROW({
    count_if(snp, v.rbegin(), v.rend(), [=](float v1) { return v1 < 0.5; });
  });
}
